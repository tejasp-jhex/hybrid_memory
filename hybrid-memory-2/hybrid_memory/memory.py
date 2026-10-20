import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import config, llm
from .embeddings import embed_documents, embed_query
from .graph_store import GraphStore, sanitize_relation
from .resolver import EntityResolver
from .vector_store import VectorStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _recency(ts: str) -> float:
    """1.0 for brand-new, decaying toward 0.5 with a configurable half-life."""
    try:
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 86400
    except Exception:
        return 1.0
    return 0.5 + 0.5 * 0.5 ** (max(age_days, 0.0) / config.RECENCY_HALF_LIFE_DAYS)


def _fact_key(f: dict) -> tuple:
    return (f["subject"], f["relation"], f["object"])


def _fact_line(f: dict) -> str:
    day = (f.get("updated") or "")[:10]
    return f"* {f['subject']} {f['relation']} {f['object']}" + (f" (noted {day})" if day else "")


class HybridMemory:
    """Graph (exact facts) + Vector (episodic recall), driven by a Haiku router.

    v2: always-on core memory, session/greeting support, graph-guided + recency-ranked
    episode retrieval, near-duplicate suppression, FORGET, and Haiku reflection.
    """

    def __init__(self, user_id: str = config.USER_ID):
        self.user_id = user_id
        self.graph = GraphStore()
        self.vectors = VectorStore()
        self.graph.ensure_user(user_id)
        self.resolver = EntityResolver(self.graph, self.vectors, user_id)
        self.dirty = False  # something changed since the last reflection

    # --------------------------------------------------------------- session
    def start_session(self) -> dict:
        meta = self.graph.get_user_meta(self.user_id)
        gap = "none (this is the first conversation)"
        if meta.get("last_seen"):
            secs = (datetime.now(timezone.utc) - datetime.fromisoformat(meta["last_seen"])).total_seconds()
            if secs < 3600:
                gap = "less than an hour ago"
            elif secs < 86400:
                gap = f"{int(secs // 3600)} hours ago"
            else:
                gap = f"{int(secs // 86400)} days ago"
        self.graph.touch_session(self.user_id, _now())
        return {"gap": gap, "sessions": (meta.get("sessions") or 0) + 1}

    def core_text(self) -> str:
        return self._core()[0]

    def _core(self) -> tuple[str, set, set]:
        """Core memory: everything known right now. Sent on every turn (and on the greeting)."""
        meta = self.graph.get_user_meta(self.user_id)
        facts = self.graph.user_facts(self.user_id, None, config.CORE_MAX_FACTS)
        recent = self.vectors.recent_episodes(self.user_id, config.CORE_RECENT_EPISODES)
        parts = []
        if meta.get("summary"):
            parts.append(f"Profile summary:\n{meta['summary']}")
        if facts:
            parts.append("Current facts (Graph):\n" + "\n".join(_fact_line(f) for f in facts))
        if recent:
            parts.append("Recent notes:\n" + "\n".join(f"* [{r['timestamp'][:10]}]: {r['text']}" for r in recent))
        text = "[Core Memory]\n" + "\n\n".join(parts) if parts else ""
        return text, {_fact_key(f) for f in facts}, {r["text"] for r in recent}

    # ------------------------------------------------------------------ turn
    def process_turn(self, user_msg: str, recent: list[dict]) -> tuple[str, dict]:
        """Route -> write -> core + retrieve. Returns (memory_context_for_sonnet, debug_info)."""
        plan = llm.route(user_msg, recent)
        debug = {"plan": plan, "applied": []}

        # Write BEFORE reading so "I moved to NYC, find me a pizza place" sees NYC.
        if plan["graph_mutations"] or plan["narrative"].strip():
            debug["applied"] = self.write(plan)

        core, core_keys, core_texts = self._core()
        retrieved = self.retrieve(plan, user_msg, core_keys, core_texts) if plan["needs_retrieval"] else ""
        return "\n\n".join(p for p in (core, retrieved) if p), debug

    # ----------------------------------------------------------------- write
    def write(self, plan: dict) -> list[str]:
        ts = _now()
        uid = self.user_id
        applied: list[str] = []
        entity_ids: set[str] = set()
        graph_changed = False
        forgot = False

        for m in plan["graph_mutations"]:
            try:
                action = m["action"]
                rel = sanitize_relation(m["relation"])
                upsert = action == "UPSERT_RELATION"
                s = self.resolver.resolve(m.get("subject") or "User", m.get("subject_type") or "Concept", create=upsert)
                o = self.resolver.resolve(m["object"], m.get("object_type") or "Concept", create=upsert)
                label = f"{m.get('subject')} -[{rel}]-> {m['object']}"
                if not s or not o:
                    applied.append(f"SKIP   {label} (entity not in graph)")
                    continue

                if upsert:
                    entity_ids |= {s, o} - {uid}
                    superseded = self.graph.upsert_relation(
                        s, rel, o, ts, single_valued=rel in config.SINGLE_VALUED_RELATIONS
                    )
                    for old in superseded:
                        n = self.vectors.tombstone(old, uid)
                        applied.append(f"SUPERSEDE old {rel} -> {old} (tombstoned {n} vectors)")
                    applied.append(f"UPSERT {label}")
                    graph_changed = True
                elif action == "DELETE_EDGE":
                    entity_ids |= {s, o} - {uid}
                    if self.graph.invalidate_relation(s, rel, o, ts):
                        n = 0 if self.graph.has_active_incoming(o) else self.vectors.tombstone(o, uid)
                        applied.append(f"DELETE {label} (tombstoned {n} vectors)")
                        graph_changed = True
                    else:
                        applied.append(f"SKIP   {label} (no active edge)")
                elif action == "FORGET":
                    forgot = True
                    n_edges = self.graph.delete_relations(s, o)
                    if n_edges:
                        n_eps = 0 if self.graph.has_active_incoming(o) else self.vectors.delete_entity_episodes(o, uid)
                        applied.append(f"FORGET {label} (removed {n_edges} edges, {n_eps} notes)")
                        graph_changed = True
                    else:
                        applied.append(f"SKIP   {label} (nothing to forget)")
            except Exception as e:  # one bad mutation must not kill the turn
                applied.append(f"ERROR  {m}: {e}")

        narrative = "" if forgot else plan["narrative"].strip()
        if narrative:
            emb = embed_documents([narrative])[0]
            if self.vectors.find_duplicate(emb, uid):
                applied.append(f"DEDUPE (near-identical note already stored) {narrative}")
            else:
                self.vectors.add_episode(narrative, emb, uid, entity_ids, ts)
                applied.append(f"EPISODE {narrative}")
                self.dirty = True

        if graph_changed:
            self.graph.clear_summary(uid)  # a stale summary could contradict fresh facts
            self.dirty = True
        return applied

    # -------------------------------------------------------------- retrieve
    def retrieve(self, plan: dict, user_msg: str, core_keys: set = frozenset(), core_texts: set = frozenset()) -> str:
        entity_ids: list[str] = []
        for ent in plan["entities"]:
            cid = self.resolver.resolve(ent["name"], ent.get("type") or "Concept", create=False)
            if cid and cid != self.user_id and cid not in entity_ids:
                entity_ids.append(cid)

        with ThreadPoolExecutor(max_workers=2) as pool:
            g = pool.submit(self._graph_search, plan, entity_ids)
            v = pool.submit(self._vector_search, plan, user_msg, entity_ids)
            facts, episodes = g.result(), v.result()

        facts = [f for f in facts if _fact_key(f) not in core_keys]          # don't repeat core memory
        episodes = [e for e in episodes if e["text"] not in core_texts]
        return self._format_retrieved(facts, episodes)

    def _graph_search(self, plan: dict, entity_ids: list[str]) -> list[dict]:
        facts: list[dict] = []
        rels = [sanitize_relation(r) for r in plan["relations"]]
        if rels:  # relation-filtered user facts beyond what core memory already holds
            facts += self.graph.user_facts(self.user_id, rels)
        facts += self.graph.two_hop(self.user_id)
        for cid in entity_ids:
            facts += self.graph.neighborhood(cid)
        seen, unique = set(), []
        for f in facts:
            if _fact_key(f) not in seen:
                seen.add(_fact_key(f))
                unique.append(f)
        return unique

    def _vector_search(self, plan: dict, user_msg: str, entity_ids: list[str]) -> list[dict]:
        """Semantic top-K + graph-linked notes, re-ranked by similarity x recency (+ small link bonus)."""
        qv = embed_query(plan["semantic_query"].strip() or user_msg)
        cands: dict[str, dict] = {}
        for e in self.vectors.search_episodes(qv, self.user_id, config.EPISODE_TOP_K * 2):
            cands[e["text"]] = {"text": e["text"], "timestamp": e["timestamp"], "sim": 1 - e["distance"], "linked": False}
        for cid in entity_ids:
            for e in self.vectors.episodes_for_entity(cid, self.user_id):
                c = cands.setdefault(
                    e["text"], {"text": e["text"], "timestamp": e["timestamp"], "sim": _cos(qv, e["embedding"]), "linked": True}
                )
                c["linked"] = True
        for c in cands.values():
            c["score"] = c["sim"] * _recency(c["timestamp"]) + (0.10 if c["linked"] else 0.0)
        ranked = sorted(cands.values(), key=lambda c: c["score"], reverse=True)
        return ranked[: config.EPISODE_TOP_K]

    @staticmethod
    def _format_retrieved(facts: list[dict], episodes: list[dict]) -> str:
        if not facts and not episodes:
            return ""
        lines = ["[Retrieved Memory]"]
        if facts:
            lines.append("Related Facts (Graph):")
            lines += [_fact_line(f) for f in facts]
        if episodes:
            lines.append("Relevant Past Notes (Vector):")
            lines += [f"* [{e['timestamp'][:10]}]: {e['text']}" for e in episodes]
        return "\n".join(lines)

    # ------------------------------------------------------------ reflection
    def consolidate(self) -> str | None:
        """Haiku condenses current facts + notes into a short profile used by core memory next session."""
        facts = self.graph.user_facts(self.user_id, None, 100)
        notes = self.vectors.recent_episodes(self.user_id, 30)
        if not facts and not notes:
            return None
        summary = llm.summarize_profile(
            "\n".join(_fact_line(f) for f in facts),
            "\n".join(f"[{n['timestamp'][:10]}] {n['text']}" for n in notes),
        )
        if summary:
            self.graph.set_summary(self.user_id, summary)
        self.dirty = False
        return summary

    # ----------------------------------------------------------------- admin
    def wipe(self):
        """Wipes this user's edges and episodes (shared concept labels are kept)."""
        self.graph.wipe_user(self.user_id)
        self.vectors.wipe_user(self.user_id)
        self.graph.ensure_user(self.user_id)
        self.dirty = False
