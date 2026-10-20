"""Local Chroma collections.

episodes        narrative snippets ("User switched from tennis to cricket.")
concept_labels  entity names/aliases, used for fuzzy entity resolution

Chroma metadata must be flat, so entity links are stored as boolean keys
`e_<entity_id>: True` and filtered with where={"e_sport_tennis": True}.
"""
import uuid

import chromadb
from chromadb.config import Settings

from . import config


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        cos = {"hnsw:space": "cosine"}
        self.episodes = self.client.get_or_create_collection("episodes", metadata=cos)
        self.concepts = self.client.get_or_create_collection("concept_labels", metadata=cos)

    # ---------- episodes ----------
    def add_episode(self, text: str, embedding: list[float], user_id: str, entity_ids: set[str], ts: str):
        meta = {"user_id": user_id, "timestamp": ts, "is_superseded": False}
        meta.update({f"e_{eid}": True for eid in entity_ids})
        self.episodes.add(
            ids=[str(uuid.uuid4())], documents=[text], embeddings=[embedding], metadatas=[meta]
        )

    def search_episodes(self, query_emb: list[float], user_id: str, k: int) -> list[dict]:
        total = self.episodes.count()
        if total == 0:
            return []
        res = self.episodes.query(
            query_embeddings=[query_emb],
            n_results=min(k, total),
            where={"$and": [{"user_id": user_id}, {"is_superseded": False}]},
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist <= config.EPISODE_MAX_DISTANCE:
                out.append({"text": doc, "timestamp": meta.get("timestamp", ""), "distance": dist})
        return out

    def tombstone(self, entity_id: str, user_id: str) -> int:
        """Soft-delete episodes that mention an entity whose fact was invalidated."""
        res = self.episodes.get(
            where={"$and": [{"user_id": user_id}, {f"e_{entity_id}": True}, {"is_superseded": False}]},
            include=["metadatas"],
        )
        for eid, meta in zip(res["ids"], res["metadatas"]):
            meta["is_superseded"] = True
            self.episodes.update(ids=[eid], metadatas=[meta])
        return len(res["ids"])

    def wipe_user(self, user_id: str):
        self.episodes.delete(where={"user_id": user_id})

    # ---------- concept labels ----------
    def add_concept_label(self, concept_id: str, label: str, ctype: str, embedding: list[float], slug: str):
        self.concepts.upsert(
            ids=[f"{concept_id}::{slug}"],
            documents=[label],
            embeddings=[embedding],
            metadatas=[{"concept_id": concept_id, "type": ctype}],
        )

    def nearest_concept(self, embedding: list[float], ctype: str) -> tuple[str, float] | None:
        if self.concepts.count() == 0:
            return None
        res = self.concepts.query(
            query_embeddings=[embedding], n_results=1, where={"type": ctype}, include=["metadatas", "distances"]
        )
        if not res["ids"][0]:
            return None
        return res["metadatas"][0][0]["concept_id"], 1.0 - res["distances"][0][0]
