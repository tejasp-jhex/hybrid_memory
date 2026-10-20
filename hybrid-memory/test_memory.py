"""End-to-end scenario test (uses Haiku + Neo4j + Chroma + Ollama, not Sonnet).   python test_memory.py"""
from hybrid_memory import HybridMemory

mem = HybridMemory(user_id="test_user")
mem.wipe()

turns = [
    "I play tennis every weekend and I live in London.",
    "I stopped playing tennis, now I'm really into cricket.",
    "I just moved to New York.",
]
recent: list[dict] = []
for t in turns:
    _, info = mem.process_turn(t, recent)
    print(f"\nUSER: {t}")
    for a in info["applied"]:
        print("   ", a)
    recent += [{"role": "user", "content": t}, {"role": "assistant", "content": "Got it."}]

active = {(f["relation"], f["object"].lower()) for f in mem.graph.user_facts("test_user", None)}
print("\nActive facts:", sorted(active))

checks = {
    "plays cricket": ("PLAYS", "cricket") in active,
    "no longer plays tennis": ("PLAYS", "tennis") not in active,
    "lives in New York": ("LIVES_IN", "new york") in active,
    "no longer lives in London": ("LIVES_IN", "london") not in active,
}

query = "Can you recommend a gear shop for my favorite sport?"
ctx, info = mem.process_turn(query, recent)
print(f"\nUSER: {query}\n{ctx}")
checks["retrieval surfaces cricket"] = "cricket" in ctx.lower()
checks["retrieval hides tennis facts"] = "PLAYS tennis" not in ctx

print()
for name, ok in checks.items():
    print(("PASS" if ok else "FAIL"), name)
raise SystemExit(0 if all(checks.values()) else 1)
