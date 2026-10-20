# Hybrid Graph + Vector Memory (Haiku · Neo4j · Chroma · nomic-embed-text · Sonnet)

```
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
1. `I play tennis and live in London` → `/memory`
2. `I stopped playing tennis, now I'm into cricket` → `/memory` (tennis edge shows as ENDED)
3. `I moved to New York` → London edge ENDED automatically (LIVES_IN is single-valued)
4. `Recommend a gear shop for my favorite sport` → turn on `/debug` to see the graph + vector context

Inspect in Neo4j Browser: `MATCH (a)-[r]->(b) RETURN a,r,b`

## Design notes
- **One Haiku call per turn** does routing, extraction and retrieval planning (forced tool call → guaranteed JSON).
- **Writes happen before retrieval**, so a message that updates a fact and asks a question sees the new fact.
- **Soft invalidation**: edges get `active=false` + `ended_at`; history is kept, retrieval ignores it.
- **Vector tombstoning**: episodes tagged with an invalidated entity get `is_superseded=true`.
  Entity tags are stored as `e_<id>: True` metadata keys since Chroma metadata must be flat.
- **Entity resolution**: alias lookup → nomic similarity (same entity type, ≥ 0.88) → new node.
  Read-path lookups never create nodes or aliases.
- Single-valued relations (`LIVES_IN`, `WORKS_AT`, ...) are configured in `config.py`.

## Tuning / known limits
- `ALIAS_SIM_THRESHOLD` (0.88) and `EPISODE_MAX_DISTANCE` (0.50) are starting points. nomic similarities on
  short labels vary, so print a few values and adjust.
- Entity matching filters by type, so inconsistent types from Haiku can cause duplicate nodes.
  The prompt pushes a fixed type vocabulary; extend it if you see duplicates.
- Tombstoning is coarse: an old episode mentioning both a deleted and a still-valid entity is hidden entirely.
- Memory write happens synchronously (~1 Haiku call + 1–2 embeddings). To cut latency, run `write()` in a
  background thread after retrieval.
- Relation names are interpolated into Cypher after strict sanitization to `[A-Z0-9_]`.
