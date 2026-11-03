# RAG Pipeline

A Retrieval-Augmented Generation (RAG) project built with LangChain, Ollama and ChromaDB. It reads PDF files, splits them into chunks, embeds them with a local Ollama embedding model, stores them in a persistent Chroma collection, retrieves the most relevant chunks with **hybrid search (vector + BM25 + Reciprocal Rank Fusion)**, and generates a grounded answer with a local Ollama LLM.

> **Status:** working end-to-end. Ingestion, hybrid retrieval, side-by-side comparison and LLM answer generation are all implemented.

## How it works


| Stage | Class / function | Details |
|-------|------------------|---------|
| Load | `get_all_documents()` | Reads every `.pdf` in a folder with `PyPDFLoader` |
| Chunk | `chunk_creator()` | `RecursiveCharacterTextSplitter`, 500 chars per chunk, 100 overlap |
| Embed | `EmbeddingManager` | Ollama `qwen3-embedding` model |
| Store | `VectorStoreManager` | Persistent Chroma collection `pdf_documents`, **`hnsw:space = cosine`**, upsert with deterministic IDs |
| Retrieve | `HybridRetriever` | Three modes: `vector`, `bm25`, `hybrid` (RRF). `compare()` shows all three side-by-side with scores |
| Answer | `run_rag()` | Builds a context-grounded prompt and calls Ollama `llama2` |

## Retrieval modes

`HybridRetriever.retrieve(query, top_k, mode=...)` supports:

| Mode | Score returned | Direction |
|------|----------------|-----------|
| `vector` | `similarity_score = 1 - cosine_distance` (plus raw `distance`) | higher = better |
| `bm25`   | `bm25_score` — true `BM25Okapi` lexical score | higher = better |
| `hybrid` | `rrf_score` — Reciprocal Rank Fusion of the two ranked lists | higher = better |

`HybridRetriever.compare(query, top_k)` prints the three result lists side-by-side so you can see where BM25 and vector disagree and how RRF resolves it.

### Notes on scores

- `rrf_score` is **only comparable within a single query** — it's a rank-based fusion weight, not an absolute relevance number.
- `bm25_score` depends on query length and corpus statistics — also not comparable across queries.
- `similarity_score` is a true cosine similarity **only because** the collection is created with `"hnsw:space": "cosine"`. If the collection was created earlier with the default L2 metric, reindex once with `run_rag(reindex=True)`.

## Tech stack

- Python 3.10+
- [LangChain](https://python.langchain.com/) — document loaders, text splitters
- [Ollama](https://ollama.com/) — local embeddings (`qwen3-embedding`) and LLM (`llama2`)
- [ChromaDB](https://www.trychroma.com/) — persistent vector store
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — BM25 lexical retrieval
- [NumPy](https://numpy.org/) — score sorting

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and the models are pulled
ollama serve                     # in a separate terminal if not already running
ollama pull qwen3-embedding
ollama pull llama2


## Usage

1. Put your PDF files in a folder.
2. Open `RAG_pipeline.py` and set that folder in the `load_all_pdf(...)` call near the top of the file.
3. Run:

```bash
python RAG_pipeline.py
```

The vector store is created on disk (by default in `DATA/vector_store`, or in the folder given by the `VECTOR_STORE_DIR` environment variable) and reused on later runs.

## Project structure

```
.
├── RAG_pipeline.py     # loading, chunking, embedding, storage and retrieval
├── requirements.txt    # Python dependencies
├── .gitignore          # keeps the vector store, PDFs and virtualenv out of git
└── README.md
```

## Roadmap

Next - Add Contextual RAG

## Notes

- The `vector_store` folder and your PDFs are excluded from git on purpose. The vector store can be rebuilt by running the pipeline again.
