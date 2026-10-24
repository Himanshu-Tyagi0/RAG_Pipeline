# RAG Pipeline

A small Retrieval-Augmented Generation (RAG) project built with LangChain, Hugging Face embeddings and ChromaDB. It reads PDF files, splits them into chunks, turns the chunks into vector embeddings, stores them in a persistent vector database, and retrieves the chunks most relevant to a question.

> **Status:** learning / practice project. Ingestion and retrieval are implemented; connecting an LLM to generate the final answer is the next step.

## How it works

```
PDFs  ->  load pages  ->  split into chunks  ->  embed  ->  ChromaDB  ->  retrieve top-k chunks
```

| Stage | Class / function | Details |
|-------|------------------|---------|
| Load | `load_all_pdf()` | Reads every `.pdf` in a folder with `PyPDFLoader` |
| Chunk | `chunk_maker()` | `RecursiveCharacterTextSplitter`, 1000 characters per chunk, 100 overlap |
| Embed | `EmbeddingManager` | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors) |
| Store | `VectorStoreManager` | Persistent ChromaDB collection named `pdf_documents` |
| Retrieve | `RAGRetriever` | Embeds the query and returns the top-k closest chunks with similarity scores |

## Tech stack

- Python 3.10+
- [LangChain](https://python.langchain.com/) (document loaders, text splitters)
- [Hugging Face sentence-transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) as the vector store

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
```

The embedding model is downloaded automatically from Hugging Face the first time you run the script.

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

- [ ] Connect an LLM (for example via Ollama or an API) to generate answers from the retrieved chunks
- [ ] Skip PDFs that are already indexed so re-runs don't re-embed them
- [ ] Add a simple command-line or web interface for asking questions
- [ ] Add tests

## Notes

- The `vector_store` folder and your PDFs are excluded from git on purpose. The vector store can be rebuilt by running the pipeline again.
