"""Entity & alias resolution: stops 'cricket', 'cricket game', 'playing cricket' becoming 3 nodes."""
import re
from datetime import datetime, timezone

from . import config
from .embeddings import embed_label
from .graph_store import GraphStore
from .vector_store import VectorStore

USER_WORDS = {"user", "i", "me", "myself", "the user"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class EntityResolver:
    def __init__(self, graph: GraphStore, vectors: VectorStore, user_id: str):
        self.graph, self.vectors, self.user_id = graph, vectors, user_id

    def resolve(self, name: str, etype: str = "Concept", create: bool = True) -> str | None:
        norm = normalize(name)
        if norm in USER_WORDS:
            return self.user_id

        # 1. exact alias lookup
        cid = self.graph.find_by_alias(norm)
        if cid:
            return cid

        # 2. semantic lookup among existing concepts of the same type
        vec = embed_label(name)
        hit = self.vectors.nearest_concept(vec, etype)
        if hit and hit[1] >= config.ALIAS_SIM_THRESHOLD:
            cid = hit[0]
            if create:  # reads must not mutate the graph
                self.graph.add_alias(cid, norm)
                self.vectors.add_concept_label(cid, name, etype, vec, slug(name))
            return cid

        if not create:
            return None

        # 3. brand-new concept
        cid = f"{slug(etype)}_{slug(name)}"
        ts = datetime.now(timezone.utc).isoformat()
        self.graph.create_concept(cid, name.strip(), etype, ts)
        self.graph.add_alias(cid, norm)
        self.vectors.add_concept_label(cid, name, etype, vec, slug(name))
        return cid
