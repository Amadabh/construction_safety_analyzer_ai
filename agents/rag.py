import sys
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Detection, Regulation
from config import Config
from qdrant_client import QdrantClient, models
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from fastembed import SparseTextEmbedding

# ---------------------------------------------------------------------------
# Safety Query Mapping
# ---------------------------------------------------------------------------

SAFETY_QUERY_MAP = {
    # Violations
    "NO-Hardhat":       "hard hat head protection requirement construction site",
    "NO-Mask":          "respiratory protection mask requirement airborne contaminants",
    "NO-Safety Vest":   "high visibility safety vest apparel requirement flagger",
    # PPE present
    "Hardhat":          "hard hat head protection compliance standard",
    "Mask":             "respiratory protection mask compliance",
    "Safety Vest":      "high visibility safety apparel compliance",
    "Gloves":           "hand protection gloves requirement construction",
    # People
    "Person":           "worker safety requirement construction site general",
    # Heavy machinery
    "Excavator":        "excavator heavy equipment operator safety clearance zone",
    "Wheel Loader":     "wheel loader heavy machinery operator safety requirement",
    "Machinery":        "machinery equipment safety requirement construction",
    "Dump Truck":       "dump truck vehicle safety construction site",
    "Truck":            "truck vehicle safety construction site",
    "Truck and Trailer":"truck trailer vehicle safety construction site",
    "Trailer":          "trailer vehicle safety requirement",
    "Semi":             "semi truck heavy vehicle safety requirement",
    # Civilian vehicles
    "SUV":              "vehicle traffic control construction zone safety",
    "Van":              "van vehicle traffic control construction zone",
    "Mini-Van":         "vehicle traffic control construction zone safety",
    "Sedan":            "vehicle traffic control construction zone safety",
    "Bus":              "bus vehicle traffic control construction zone",
    "Vehicle":          "vehicle traffic control construction zone safety",
    # Site objects
    "Safety Cone":      "traffic cone safety barrier construction zone requirement",
    "Ladder":           "ladder safety requirement portable climbing construction",
    "Fire Hydrant":     "fire hydrant clearance requirement obstruction",
}

VIOLATION_LABELS = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

PRIORITY_MAP = {
    "HIGH":   {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"},
    "MEDIUM": {"Excavator", "Wheel Loader", "Machinery", "Dump Truck", "Ladder"},
    "LOW":    {
        "Person", "Safety Cone", "SUV", "Van", "Sedan", "Bus",
        "Truck", "Semi", "Trailer", "Truck and Trailer", "Mini-Van",
        "Vehicle", "Fire Hydrant", "Hardhat", "Mask", "Safety Vest", "Gloves",
    },
}

RETRIEVAL_LIMITS = {"HIGH": 5, "MEDIUM": 3, "LOW": 1}
CONFIDENCE_THRESHOLD = 0.50
SCORE_THRESHOLD      = 0.60


def get_priority(label: str) -> str:
    for level, labels in PRIORITY_MAP.items():
        if label in labels:
            return level
    return "LOW"


def build_query(label: str) -> str:
    return SAFETY_QUERY_MAP.get(label, f"{label} safety requirement construction site")


# ---------------------------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    def __init__(self):
        self.client          = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        self.dense_embeddings  = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_embeddings = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        self.collection_name = Config.QDRANT_COLLECTION

    def _embed(self, query_text: str):
        """Return both dense and sparse vectors for a query string."""
        dense  = self.dense_embeddings.embed_query(query_text)
        sparse = list(self.sparse_embeddings.embed([query_text]))[0]
        return dense, sparse

    def _search(self, query_text: str, limit: int) -> list:
        """Hybrid search using dense + sparse vectors fused with RRF."""
        dense_vec, sparse_vec = self._embed(query_text)

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=limit * 3,        # over-fetch before fusion
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    ),
                    using="sparse",
                    limit=limit * 3,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            score_threshold=SCORE_THRESHOLD,
        )
        return results.points if hasattr(results, "points") else results

    def retrieve_regulations(self, context: list[Detection]) -> list[Regulation]:
        """Retrieve relevant OSHA regulations based on detected violations."""

        if not context:
            return []

        print(f"Retrieving regulations for {len(context)} detections...")

        regulations_dict = {}   # deduplicate by (text hash) so same regulation from
                                # different labels doesn't appear twice
        query_cache      = {}   # avoid re-embedding the same query string

        for detection in context:
            # Skip low-confidence detections
            if detection.confidence < CONFIDENCE_THRESHOLD:
                continue

            # Skip LOW-priority non-violations to save compute
            priority = get_priority(detection.label)
            if priority == "LOW" and detection.label not in VIOLATION_LABELS:
                continue

            query_text = build_query(detection.label)
            limit      = RETRIEVAL_LIMITS.get(priority, 1)

            # Use cached results if same query was already run
            if query_text in query_cache:
                points = query_cache[query_text]
            else:
                points = self._search(query_text, limit)
                query_cache[query_text] = points

            for point in points:
                payload   = point.payload or {}
                text      = payload.get("text", "")
                text_key  = hash(text[:200])        # deduplicate by content

                if text_key not in regulations_dict:
                    regulations_dict[text_key] = Regulation(
                        citation=payload.get("source", "Unknown"),
                        text=text,
                        source="CAL_OSHA",
                    )

        regulations = list(regulations_dict.values())
        print(f"✓ Retrieved {len(regulations)} relevant regulations")
        return regulations
    


if __name__ == "__main__":
    retriever = RAGRetriever()

    # Simulate a frame with mixed detections
    test_detections = [
        Detection(label="NO-Hardhat",     confidence=0.91, bbox=[100, 50, 60, 80]),
        Detection(label="NO-Safety Vest", confidence=0.85, bbox=[200, 100, 70, 90]),
        Detection(label="Excavator",      confidence=0.78, bbox=[300, 200, 200, 150]),
        Detection(label="Sedan",          confidence=0.72, bbox=[400, 300, 100, 60]),
        Detection(label="Ladder",         confidence=0.65, bbox=[150, 120, 40, 100]),
    ]

    regulations = retriever.retrieve_regulations(test_detections)

    print(f"\n{'='*60}")
    for reg in regulations:
        print(f"\nSource  : {reg.citation}")
        print(f"Text    : {reg.text[:200]}...")
    print(f"\n{'='*60}")
    print(f"Total regulations retrieved: {len(regulations)}")