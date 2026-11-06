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

The vector store is created on dis# RAG Pipeline

A Retrieval-Augmented Generation (RAG) project built with LangChain, Ollama and ChromaDB. It reads PDF files, splits them into chunks, embeds them with a local Ollama embedding model, stores them in a persistent Chroma collection, retrieves the most relevant chunks with **hybrid search (vector + BM25 + Reciprocal Rank Fusion)**, and generates a grounded answer with a local Ollama LLM.

> **Status:** working end-to-end. Ingestion, hybrid retrieval, side-by-side comparison and LLM answer generation are all implemented. Retrieval quality is measured on a 26-query golden dataset.

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

## Evaluation

Retrieval quality is measured on **26 handcrafted queries** with manually-verified ground-truth relevant chunks, built from the "Attention Is All You Need" paper (86 chunks).

Metrics at **K=5**:

| Mode | Recall@5 | Precision@5 | MRR |
|------|----------|-------------|-----|
| Vector only | 0.846 | 0.277 | 0.667 |
| BM25 only | 0.718 | 0.231 | 0.574 |
| **Hybrid (RRF, 0.5/0.5)** | **0.865** | **0.285** | **0.703** |

**Hybrid search outperforms both individual retrievers** on all three metrics. Vector-only is strong but misses keyword-specific queries; BM25-only fails on semantic queries where the answer doesn't contain the query's exact words. Hybrid balances both, improving **MRR by +3.6 points over vector-only** — meaning the most relevant chunk appears higher in the ranking.

### Weight sweep

To check whether the default fusion weights are optimal, we swept BM25/vector weights:

| (bm25, vec) | Recall@5 | Precision@5 | MRR |
|-------------|----------|-------------|-----|
| **(0.5, 0.5)** | **0.865** | **0.285** | 0.703 |
| (0.4, 0.6) | 0.846 | 0.277 | **0.721** |
| (0.3, 0.7) | 0.846 | 0.277 | 0.715 |
| (0.6, 0.4) | 0.718 | 0.231 | 0.662 |
| (0.7, 0.3) | 0.718 | 0.231 | 0.667 |

The default **`(0.5, 0.5)` achieves the best Recall@5 and Precision@5**. `(0.4, 0.6)` trades 2 points of recall for ~2 points of MRR — within noise on a 26-query set. We kept `(0.5, 0.5)` as the default to avoid overfitting to the evaluation set.

### Critical bug found during evaluation

Initial BM25 implementation used naive whitespace tokenization (`text.lower().split()`), which treated `"decoder:"` (with trailing colon) as a **different token** from `"decoder"` (the query term). This caused BM25 to **underperform vector-only search** (Recall@5: 0.513 vs 0.846) and made hybrid **worse** than vector alone.

Switching to regex-based tokenization — `re.findall(r"\b\w+\b", text.lower())` — which strips punctuation and matches word boundaries raised **BM25 recall by +20 points** (0.513 → 0.718) and turned hybrid into the clear winner (recall **0.865**). This illustrates why tokenization is critical in lexical retrieval systems and why evaluation is the only way to catch such issues.

### Evaluation methodology

- **Queries:** 26 handcrafted, spanning architecture, attention mechanisms, training, results, and complexity
- **Ground truth:** 1–3 manually-verified relevant chunks per query (verified against paper sections, noise chunks pruned)
- **Metrics:** Recall@K (coverage of relevant chunks), Precision@K (density of relevant chunks in top-K), MRR (mean reciprocal rank of first relevant chunk)
- **Reproducibility:** `dump_chunks.py` exports chunks with stable content-hash IDs; `evaluate.py` runs the benchmark

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
git clone https://github.com/Himanshu-Tyagi0/RAG_Pipeline.git
cd RAG_Pipeline

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and the models are pulled
ollama serve                     # in a separate terminal if not already running
ollama pull qwen3-embedding
ollama pull llama2k (by default in `DATA/vector_store`, or in the folder given by the `VECTOR_STORE_DIR` environment variable) and reused on later runs.

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
