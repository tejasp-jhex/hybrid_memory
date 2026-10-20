import os
from dotenv import load_dotenv

load_dotenv()

HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-5-5")
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-5-5")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "memorypass123")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
USER_ID = os.getenv("MEMORY_USER_ID", "tejas")

# Cosine similarity needed to treat a new entity name as an alias of an existing concept
ALIAS_SIM_THRESHOLD = float(os.getenv("ALIAS_SIM_THRESHOLD", "0.88"))
# Episodic (narrative) retrieval
EPISODE_TOP_K = int(os.getenv("EPISODE_TOP_K", "3"))
EPISODE_MAX_DISTANCE = float(os.getenv("EPISODE_MAX_DISTANCE", "0.50"))  # cosine distance

# Relations where a user can only have one active value at a time.
# A new UPSERT automatically invalidates the old edge.
SINGLE_VALUED_RELATIONS = {"LIVES_IN", "WORKS_AT", "HAS_NAME", "HAS_JOB_TITLE"}
