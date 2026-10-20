"""Neo4j knowledge graph.

Schema
  (:Entity:User    {id, name})
  (:Entity:Concept {id, name, type})          e.g. id = "sport_cricket"
  (:Alias {name})-[:ALIAS_OF]->(:Concept)     normalized surface forms
  (a)-[:PLAYS {active, created_at, updated_at, ended_at}]->(b)

Edges are never physically deleted: invalidated edges get active=false + ended_at,
so history is preserved but retrieval only sees active facts.
"""
import re

from neo4j import GraphDatabase

from . import config


def sanitize_relation(rel: str) -> str:
    rel = re.sub(r"[^A-Z0-9_]", "", rel.upper().replace(" ", "_").replace("-", "_"))
    if not rel:
        rel = "RELATED_TO"
    if rel[0].isdigit():
        rel = "R_" + rel
    return rel


class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self.driver.verify_connectivity()
        self._init_schema()

    def close(self):
        self.driver.close()

    def _run(self, query: str, **params) -> list[dict]:
        with self.driver.session() as s:
            return s.run(query, **params).data()

    def _init_schema(self):
        self._run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        self._run("CREATE CONSTRAINT alias_name IF NOT EXISTS FOR (a:Alias) REQUIRE a.name IS UNIQUE")

    # ---------- nodes ----------
    def ensure_user(self, user_id: str):
        self._run(
            "MERGE (u:Entity:User {id:$id}) ON CREATE SET u.name='User', u.type='User'", id=user_id
        )

    def find_by_alias(self, norm_name: str) -> str | None:
        rows = self._run(
            "MATCH (:Alias {name:$n})-[:ALIAS_OF]->(c:Concept) RETURN c.id AS id LIMIT 1", n=norm_name
        )
        return rows[0]["id"] if rows else None

    def create_concept(self, cid: str, name: str, ctype: str, ts: str):
        self._run(
            "MERGE (c:Entity:Concept {id:$id}) ON CREATE SET c.name=$name, c.type=$type, c.created_at=$ts",
            id=cid, name=name, type=ctype, ts=ts,
        )

    def add_alias(self, cid: str, norm_alias: str):
        self._run(
            "MATCH (c:Concept {id:$cid}) MERGE (a:Alias {name:$alias}) MERGE (a)-[:ALIAS_OF]->(c)",
            cid=cid, alias=norm_alias,
        )

    # ---------- edges ----------
    def upsert_relation(self, s: str, rel: str, o: str, ts: str, single_valued: bool) -> list[str]:
        """Create/reactivate edge. Returns ids of objects whose edge was superseded."""
        superseded: list[str] = []
        if single_valued:
            rows = self._run(
                f"""MATCH (a:Entity {{id:$s}})-[r:{rel}]->(b:Entity)
                    WHERE r.active = true AND b.id <> $o
                    SET r.active = false, r.ended_at = $ts
                    RETURN b.id AS id""",
                s=s, o=o, ts=ts,
            )
            superseded = [r["id"] for r in rows]
        self._run(
            f"""MATCH (a:Entity {{id:$s}}), (b:Entity {{id:$o}})
                MERGE (a)-[r:{rel}]->(b)
                ON CREATE SET r.created_at = $ts
                SET r.active = true, r.updated_at = $ts
                REMOVE r.ended_at""",
            s=s, o=o, ts=ts,
        )
        return superseded

    def invalidate_relation(self, s: str, rel: str, o: str, ts: str) -> int:
        rows = self._run(
            f"""MATCH (a:Entity {{id:$s}})-[r:{rel}]->(b:Entity {{id:$o}})
                WHERE r.active = true
                SET r.active = false, r.ended_at = $ts, r.updated_at = $ts
                RETURN count(r) AS n""",
            s=s, o=o, ts=ts,
        )
        return rows[0]["n"]

    def has_active_incoming(self, entity_id: str) -> bool:
        rows = self._run(
            "MATCH ()-[r]->(e:Entity {id:$id}) WHERE r.active = true RETURN count(r) AS n", id=entity_id
        )
        return rows[0]["n"] > 0

    # ---------- reads ----------
    def user_facts(self, user_id: str, rels: list[str] | None, limit: int = 30) -> list[dict]:
        return self._run(
            """MATCH (u:Entity {id:$uid})-[r]->(c:Entity)
               WHERE r.active = true AND ($rels IS NULL OR type(r) IN $rels)
               RETURN u.name AS subject, type(r) AS relation, c.name AS object, r.updated_at AS updated
               ORDER BY r.updated_at DESC LIMIT $limit""",
            uid=user_id, rels=rels or None, limit=limit,
        )

    def two_hop(self, user_id: str, limit: int = 15) -> list[dict]:
        return self._run(
            """MATCH (u:Entity {id:$uid})-[r1]->(c:Entity)-[r2]->(d:Entity)
               WHERE r1.active = true AND r2.active = true AND d.id <> $uid
               RETURN c.name AS subject, type(r2) AS relation, d.name AS object, r2.updated_at AS updated
               ORDER BY r2.updated_at DESC LIMIT $limit""",
            uid=user_id, limit=limit,
        )

    def neighborhood(self, entity_id: str, limit: int = 15) -> list[dict]:
        return self._run(
            """MATCH (e:Entity {id:$id})-[r]-(o:Entity)
               WHERE r.active = true
               RETURN startNode(r).name AS subject, type(r) AS relation, endNode(r).name AS object,
                      r.updated_at AS updated
               ORDER BY r.updated_at DESC LIMIT $limit""",
            id=entity_id, limit=limit,
        )

    def dump(self, user_id: str) -> list[dict]:
        """Everything incl. invalidated history, for debugging."""
        return self._run(
            """MATCH (a:Entity)-[r]->(b:Entity)
               WHERE a.id = $uid OR b.id = $uid OR EXISTS { MATCH (:Entity {id:$uid})-[]->(a) }
               RETURN a.name AS subject, type(r) AS relation, b.name AS object,
                      r.active AS active, r.created_at AS created, r.ended_at AS ended
               ORDER BY r.created_at""",
            uid=user_id,
        )

    def wipe_user(self, user_id: str):
        self._run("MATCH (u:Entity {id:$uid}) DETACH DELETE u", uid=user_id)
