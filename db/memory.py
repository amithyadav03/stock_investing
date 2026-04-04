import chromadb
from core.config import settings

# Initialize ChromaDB client pointing to the defined local storage
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

# Get or create the 'experiences' collection (for Reinforcement Learning Memory)
# using Default embedding function (all-MiniLM-L6-v2) under the hood
experience_collection = chroma_client.get_or_create_collection(
    name="trade_experiences",
    metadata={"hnsw:space": "cosine"}
)

def add_experience(trade_id: str, document: str, metadatas: dict):
    """
    Store a past trade along with its analysis, outcome, and PnL.
    document: A rich string combining the rationale and the outcome (e.g. "Trade XYZ bought at 100, exited at 90 due to RBI rate hike.")
    metadatas: Tags for filtering e.g. {"symbol": "RELIANCE", "direction": "BUY", "outcome": "LOSS"}
    """
    experience_collection.add(
        documents=[document],
        metadatas=[metadatas],
        ids=[trade_id]
    )

def retrieve_similar_experiences(query_text: str, n_results: int = 3):
    """
    Search for past similar market setups or stock rationale.
    """
    results = experience_collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results
