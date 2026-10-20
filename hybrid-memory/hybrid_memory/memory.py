from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import config, llm
from .embeddings import embed_documents, embed_query
from .graph_store import GraphStore, sanitize_relation
from .resolver import EntityResolver
from .vector_store import VectorStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HybridMemory:
    """Graph (exact facts) + Vector (episodic recall), driven by a Haiku router."""

    def __init__(self, user_id: str = config.USER_ID):
        self.user_id = user_id
        self.graph = GraphStore()
        self.vectors = VectorStore()
        self.graph.ensure_user(user_id)
        self.resolver = EntityResolver(self.graph, self.vectors, user_id)

    # ------------------------------------------------------------------ turn
    def process_turn(self, user_msg: str, recent: list[dict]) -> tuple[str, dict]:
        """Route -> write -> retrieve. Returns (memory_context_for_sonnet, debug_info)."""
        plan = llm.route(user_msg, recent)
        debug = {"plan": plan, "applied": []}

        # Write BEFORE retrieve so "I moved to NYC, find me a pizza place" sees NYC.
        if plan["graph_mutations"] or plan["narrative"].strip():
            debug["applied"] = self.write(plan)

        context = self.retrieve(plan, user_msg) if plan["needs_retrieval"] else ""
        return context, debug

    # ----------------------------------------------------------------- write
    def write(self, plan: dict) -> list[str]:
        ts = _now()
        applied: list[str] = []
        entity_ids: set[str] = set()

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
                entity_ids |= {s, o} - {self.user_id}

                if upsert:
                    superseded = self.graph.upsert_relation(
                        s, rel, o, ts, single_valued=rel in config.SINGLE_VALUED_RELATIONS
                    )
                    for old in superseded:
                        n = self.vectors.tombstone(old, self.user_id)
                        applied.append(f"SUPERSEDE old {rel} -> {old} (tombstoned {n} vectors)")
                    applied.append(f"UPSERT {label}")
                else:
                    if self.graph.invalidate_relation(s, rel, o, ts):
                        n = 0
                        if not self.graph.has_active_incoming(o):
                            n = self.vectors.tombstone(o, self.user_id)
                        applied.append(f"DELETE {label} (tombstoned {n} vectors)")
                    else:
                        applied.append(f"SKIP   {label} (no active edge)")
            except Exception as e:  # one bad mutation must not kill the turn
                applied.append(f"ERROR  {m}: {e}")

        narrative = plan["narrative"].strip()
        if narrative:
            self.vectors.add_episode(narrative, embed_documents([narrative])[0], self.user_id, entity_ids, ts)
            applied.append(f"EPISODE {narrative}")
        return applied

    # -------------------------------------------------------------- retrieve
    def retrieve(self, plan: dict, user_msg: str) -> str:
        with ThreadPoolExecutor(max_workers=2) as pool:
            g = pool.submit(self._graph_search, plan)
            v = pool.submit(self._vector_search, plan, user_msg)
            facts, episodes = g.result(), v.result()
        return self._format(facts, episodes)

    def _graph_search(self, plan: dict) -> list[dict]:
        rels = [sanitize_relation(r) for r in plan["relations"]] or None
        facts = self.graph.user_facts(self.user_id, rels)
        if rels and not facts:  # planner guessed a relation name that doesn't exist -> fall back to all
            facts = self.graph.user_facts(self.user_id, None)
        facts += self.graph.two_hop(self.user_id)
        for ent in plan["entities"]:
            cid = self.resolver.resolve(ent["name"], ent.get("type") or "Concept", create=False)
            if cid and cid != self.user_id:
                facts += self.graph.neighborhood(cid)
        seen, unique = set(), []
        for f in facts:
            key = (f["subject"], f["relation"], f["object"])
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def _vector_search(self, plan: dict, user_msg: str) -> list[dict]:
        query = plan["semantic_query"].strip() or user_msg
        return self.vectors.search_episodes(embed_query(query), self.user_id, config.EPISODE_TOP_K)

    @staticmethod
    def _format(facts: list[dict], episodes: list[dict]) -> str:
        if not facts and not episodes:
            return ""
        lines = ["[System Memory Context]"]
        if facts:
            lines.append("Verified User Facts (Graph):")
            lines += [f"* {f['subject']} {f['relation']} {f['object']}" for f in facts]
        if episodes:
            lines.append("Relevant Conversation History (Vector):")
            lines += [f"* [{e['timestamp'][:10]}]: {e['text']}" for e in episodes]
        return "\n".join(lines)

    # ----------------------------------------------------------------- admin
    def wipe(self):
        """Wipes this user's edges and episodes (shared concept labels are kept)."""
        self.graph.wipe_user(self.user_id)
        self.vectors.wipe_user(self.user_id)
        self.graph.ensure_user(self.user_id)
