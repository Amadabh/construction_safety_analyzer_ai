import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from config import Config

DENSE_MODEL  = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
EMBEDDING_SIZE = 384
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

SKIP_PATTERNS = [
    r'table of contents',
    r'copyright\s*©',
    r'all other rights reserved',
    r'publications unit',
    r'department of industrial relations',
    r'pocket guide',
    r'www\.dir\.ca\.gov',
    r'http://leginfo',
    r'nonprofit and educational purposes',
    r'\.\s*\.\s*\.\s*\.\s*\.',   # dot leaders like ". . . . ."
    r'^\s*\d+\s*$',              # pages that are just a number
    r'about this pocket guide',
]


def is_useful_chunk(text: str) -> bool:
    text_lower = text.lower()

    # Skip if too short to be a real regulation
    if len(text.strip()) < 100:
        return False

    # Skip if matches any noise pattern
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, text_lower):
            return False

    return True


def clean_text(text: str) -> str:
    """Fix common PDF extraction artifacts before embedding."""
    ligature_map = {
        '\ue03e': 'fl', '\ue03f': 'fi', '\ue040': 'ff',
        '\ue050': 'fi', '\ue051': 'fl', '\ue052': 'ff',
        '\ue053': 'ffi', '\ue054': 'ffl',
    }
    for char, replacement in ligature_map.items():
        text = text.replace(char, replacement)

    # Fix hyphenated line breaks e.g. "retro-\nreflective" -> "retroreflective"
    text = re.sub(r'-\n', '', text)

    # Normalize whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


# ---------------------------------------------------------------------------
# Load & Chunk
# ---------------------------------------------------------------------------

def load_and_chunk(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Skip front matter (cover, TOC, copyright)
    SKIP_PAGES = set(range(0, 5))
    documents = [doc for doc in documents if doc.metadata.get("page", 0) not in SKIP_PAGES]

    # Clean text before splitting
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    # Deduplicate
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        h = hash(chunk.page_content[:200])
        if h not in seen:
            seen.add(h)
            unique_chunks.append(chunk)

    # Filter noise
    useful_chunks = [c for c in unique_chunks if is_useful_chunk(c.page_content)]

    print(f"✓ {len(chunks)} chunks → {len(unique_chunks)} after dedup → {len(useful_chunks)} after noise filter")
    return useful_chunks


# ---------------------------------------------------------------------------
# Qdrant Setup — hybrid collection (dense + sparse)
# ---------------------------------------------------------------------------

def setup_collection(client: QdrantClient, collection_name: str):
    """Drop and recreate collection with both dense and sparse vector support."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_SIZE,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )
    print(f"✓ Created hybrid collection: {collection_name}")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest(pdf_path: str):
    chunks = load_and_chunk(pdf_path)

    # Load both embedding models
    dense_embeddings  = FastEmbedEmbeddings(model_name=DENSE_MODEL)
    sparse_embeddings = SparseTextEmbedding(model_name=SPARSE_MODEL)

    client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
    setup_collection(client, Config.QDRANT_COLLECTION)

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_end    = min(batch_start + BATCH_SIZE, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_texts  = [chunk.page_content for chunk in batch_chunks]

        # Embed — dense and sparse
        batch_dense  = dense_embeddings.embed_documents(batch_texts)
        batch_sparse = list(sparse_embeddings.embed(batch_texts))

        points = [
            models.PointStruct(
                id=batch_start + i,
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    )
                },
                payload={
                    "text":     chunk.page_content,
                    "source":   chunk.metadata.get("source", ""),
                    "page":     chunk.metadata.get("page", 0),
                    "chunk_id": batch_start + i,
                }
            )
            for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(batch_chunks, batch_dense, batch_sparse))
        ]

        client.upsert(collection_name=Config.QDRANT_COLLECTION, points=points)
        print(f"✓ Uploaded batch {batch_start}–{batch_end}")

    print(f"\n✓ Ingestion complete: {len(chunks)} chunks stored in '{Config.QDRANT_COLLECTION}'")


if __name__ == "__main__":
    ingest("./data/docs/CAL_OSHA.pdf")