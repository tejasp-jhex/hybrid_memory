# Hybrid Graph + Vector Memory (Haiku · Neo4j · Chroma · nomic-embed-text · Sonnet)

```
[session start] core memory ──► Sonnet greets first

user msg + last 5 ──► Haiku router (one tool call) ──► write plan + retrieval plan
                          │
        ┌─────────────────┴──────────────────┐
   WRITE (first)                        RETRIEVE (parallel)
   resolve entities (alias → vector)    Neo4j: user facts + 2-hop + entity neighborhoods
   Neo4j upsert / invalidate edges      Chroma: top-K episodes (superseded ones filtered out)
   Chroma: embed narrative + tag ids              │
        └──────────────► fused [System Memory Context] ──► Sonnet (streaming)
```

## Setup
```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up -d            # Neo4j  (browser: http://localhost:7474)
ollama pull nomic-embed-text    # local embeddings
pip install -r requirements.txt

python test_memory.py           # scripted tennis → cricket / London → New York scenario
python chatbot.py               # interactive; try /debug and /memory
```

## Things to try in the chatbot
- Start it: Claude **greets you first**, using core memory (name, current facts, recent notes, last-seen gap).
- `I play tennis and live in London` → `/memory`
- `I stopped playing tennis, now I'm into cricket` → tennis shows as ENDED
- `I moved to New York` → London ENDED automatically (LIVES_IN is single-valued)
- `Forget that I ever lived in London` → hard-deleted with its notes
- `/core` shows exactly what is sent to Sonnet every turn; `/debug` shows router plan + applied writes
- `/quit` runs a Haiku reflection that rebuilds the profile summary for the next greeting

Inspect in Neo4j Browser: `MATCH (a)-[r]->(b) RETURN a,r,b`

## What's new in v2
| Improvement | What it does |
| --- | --- |
| **Core memory** | All current facts + profile summary + recent notes are sent on *every* turn (and to the greeting), so Sonnet never "forgets" the basics. The per-turn retrieval block adds depth and skips anything already in core. |
| **Greets first** | Sonnet opens the chat using core memory, time of day, and how long since the last session. |
| **Graph-guided retrieval** | Notes linked to entities in the query are pulled even when embeddings alone would miss them. |
| **Recency-aware ranking** | `score = similarity × recency (half-life 90d, floor 0.5) + link bonus`. |
| **Near-duplicate suppression** | Re-stating a fact doesn't pile up identical notes. |
| **FORGET** | "Forget X" hard-deletes the edges and every note mentioning the entity. |
| **Reflection** | On exit (or `/reflect`) Haiku condenses facts + notes into a short summary. Any later graph change clears it, so a stale summary can't contradict fresh facts. |
| **Dated facts** | Facts carry a "noted <date>" stamp so Sonnet can reason about recency. |

## Design notes
- **One Haiku call per turn** does routing, extraction and retrieval planning (forced tool call → guaranteed JSON).
- **Writes happen before reads**, so a message that updates a fact and asks a question sees the new fact.
- **Soft invalidation**: ended edges get `active=false` + `ended_at`; history is kept, retrieval ignores it. FORGET is the only hard delete.
- **Vector tombstoning**: notes tagged with an invalidated entity get `is_superseded=true` (Chroma metadata must be flat, so tags are `e_<id>: True` keys).
- **Entity resolution**: alias lookup → nomic similarity (same type, ≥ 0.88) → new node. Read-path lookups never create nodes or aliases.
- Only the **user's** messages are ever written to memory, never the assistant's.

## Tuning / known limits
- `ALIAS_SIM_THRESHOLD` (0.88) and `EPISODE_MAX_DISTANCE` (0.50) are starting points; print a few real nomic similarities and adjust.
- Entity matching filters by type, so inconsistent types from Haiku can cause duplicate nodes. Extend the type vocabulary in `prompts.py` if you see them.
- Tombstoning is coarse: an old note mentioning both a deleted and a still-valid entity is hidden entirely.
- Core memory is capped at `CORE_MAX_FACTS` (60) most-recently-updated facts; older ones are still reachable through retrieval.
- Writes are synchronous (1 Haiku call + 1–2 embeddings). To cut latency, run `write()` in a background thread.
- `recent_episodes` / consolidation load a user's active notes and sort in Python. Fine for personal scale; add an indexed store if it grows to tens of thousands.
- Relation names are interpolated into Cypher after strict sanitization to `[A-Z0-9_]`.
