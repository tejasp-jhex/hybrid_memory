"""Tests for the memory system.

  python test_memory.py         deterministic tests (Neo4j + Chroma + Ollama; NO Anthropic calls)
  python test_memory.py --llm   also run the Haiku router end-to-end scenario
"""
import sys

from hybrid_memory import HybridMemory
from hybrid_memory.embeddings import embed_query
from hybrid_memory.llm import EMPTY_PLAN

UID = "test_user"
mem = HybridMemory(user_id=UID)
mem.wipe()
checks: dict[str, bool] = {}


def plan(muts, narrative=""):
    return {**EMPTY_PLAN, "graph_mutations": muts, "narrative": narrative}


def m(action, rel, obj, otype):
    return {"action": action, "subject": "User", "subject_type": "User", "relation": rel,
            "object": obj, "object_type": otype}


def active():
    return {(f["relation"], f["object"].lower()) for f in mem.graph.user_facts(UID, None)}


def episode_texts(q):
    return [e["text"] for e in mem.vectors.search_episodes(embed_query(q), UID, 10)]


# ---------------- deterministic scenario ----------------
for line in mem.write(plan([m("UPSERT_RELATION", "PLAYS", "tennis", "Sport"),
                            m("UPSERT_RELATION", "LIVES_IN", "London", "City")],
                           "User plays tennis every weekend and lives in London.")):
    print(line)
for line in mem.write(plan([m("DELETE_EDGE", "PLAYS", "tennis", "Sport"),
                            m("UPSERT_RELATION", "PLAYS", "cricket", "Sport")],
                           "User switched from tennis to cricket.")):
    print(line)
for line in mem.write(plan([m("UPSERT_RELATION", "LIVES_IN", "New York", "City")], "User moved to New York.")):
    print(line)

a = active()
checks["plays cricket"] = ("PLAYS", "cricket") in a
checks["tennis ended (not active)"] = ("PLAYS", "tennis") not in a
checks["lives in New York"] = ("LIVES_IN", "new york") in a
checks["London auto-superseded"] = ("LIVES_IN", "london") not in a
checks["history kept (ended edges exist)"] = any(not r["active"] for r in mem.graph.dump(UID))

texts = episode_texts("tennis")
checks["old tennis note tombstoned"] = not any("plays tennis every weekend" in t for t in texts)
checks["switch note still retrievable"] = any("switched from tennis to cricket" in t for t in texts)

before = mem.vectors.episodes.count()
mem.write(plan([], "User switched from tennis to cricket."))
checks["near-duplicate note skipped"] = mem.vectors.episodes.count() == before

core = mem.core_text()
checks["core memory has facts"] = "cricket" in core and "New York" in core
checks["core memory hides ended facts"] = "London" not in core.split("Recent notes")[0]

ctx = mem.retrieve({**EMPTY_PLAN, "needs_retrieval": True, "entities": [{"name": "cricket", "type": "Sport"}],
                    "semantic_query": "sports equipment"}, "gear shop?")
checks["graph-linked retrieval finds cricket note"] = "cricket" in ctx.lower()

mem.write(plan([m("FORGET", "LIVES_IN", "New York", "City")]))
checks["FORGET removes fact"] = ("LIVES_IN", "new york") not in active()
checks["FORGET removes notes"] = not any("New York" in t for t in episode_texts("moved to New York"))
checks["FORGET removes ended history"] = not any(
    r["object"] == "New York" for r in mem.graph.dump(UID))

# ---------------- optional: Haiku router end-to-end ----------------
if "--llm" in sys.argv:
    mem.wipe()
    recent: list[dict] = []
    for t in ["I play tennis every weekend and I live in London.",
              "I stopped playing tennis, now I'm really into cricket.",
              "I just moved to New York."]:
        _, info = mem.process_turn(t, recent)
        print(f"\nUSER: {t}")
        for line in info["applied"]:
            print("   ", line)
        recent += [{"role": "user", "content": t}, {"role": "assistant", "content": "Got it."}]
    a = active()
    checks["[llm] plays cricket"] = ("PLAYS", "cricket") in a
    checks["[llm] tennis ended"] = ("PLAYS", "tennis") not in a
    checks["[llm] lives in New York"] = ("LIVES_IN", "new york") in a
    ctx, _ = mem.process_turn("Can you recommend a gear shop for my favorite sport?", recent)
    checks["[llm] context mentions cricket"] = "cricket" in ctx.lower()

print()
for name, ok in checks.items():
    print(("PASS" if ok else "FAIL"), name)
mem.wipe()
raise SystemExit(0 if all(checks.values()) else 1)
