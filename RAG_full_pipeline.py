from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_core.documents import Document
from typing import Optional
import re

path = "C:\\Users\\tyagi\\langchain_projects\\RAG_Understanding"
embedding_model = "qwen3-embedding" #Ollam
llm_model = "llama2" #Ollama 


def get_all_documents(folder_path=path):
    if folder_path is None:
        folder_path = os.path.join(os.getcwd(),"Data")

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not Found : {folder_path}")

    num_docs = 0
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path,filename)

            loader = PyPDFLoader(pdf_path)

            doc = loader.load()

            all_docs.extend(doc)
            num_docs +=1
        else:
            print("No file found which endswith .pdf")

    print("Number of documents added :",num_docs)

    return all_docs



def chunk_creator(documents,chunk_size=500,chunk_overlap=100):
    chunk_generator = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunked = chunk_generator.split_documents(documents)

    return chunked

class EmbeddingManager:
    def __init__(self):
        self.model = OllamaEmbeddings(
            model=embedding_model
        )

    def embedding_generator(self,text):
        embedding = self.model.embed_documents(texts=text)
        return embedding


class VectorStoreManager:
    def __init__(
        self,
        persists_directory=None,
        collection_name="pdf_documents"
    ):
        self.persists_directory = persists_directory
        self.collection_name = collection_name
        self.collection = None
        self.client = None

        self._initialize_store()

    def _initialize_store(self):
        if self.persists_directory is None:
            self.persists_directory = os.path.abspath("vector_store")
            os.makedirs(self.persists_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persists_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "This is demo RAG",
                      "hnsw:space" : "cosine"},
        )

    def add_documents(self, embeddings, documents, ids):
        if not (len(embeddings) == len(documents) == len(ids)):
            raise ValueError(
                "Number of documents, embeddings and ids must match"
            )

        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):
            metadata = dict(doc.metadata or {})
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            all_metadata.append(metadata)

            documents_content.append(doc.page_content)

            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            embeddings_list.append(embedding)

        if not ids:
            print("No documents to add.")
            return

        try:
            # upsert + deterministic ids => re-running never creates duplicates
            self.collection.upsert(
                ids=ids,
                metadatas=all_metadata,
                documents=documents_content,
                embeddings=embeddings_list
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add documents  to Chroma : {e}") from e

        print(
            "Total documents added to Vector Store:",
            len(documents_content)
        )
        print("Docs in collection:", self.collection.count())


class HybridRetriever:
    RRF_K = 60

    def __init__(self, embedding_manager, vector_store,
                weights=(0.5, 0.5), vector_k=5, bm25_k=5):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.weights = list(weights)
        self.vector_k = vector_k
        self.bm25_k = bm25_k

        self.bm25_docs: list[Document] = []
        self.bm25: BM25Okapi | None = None

        # ← this is the ONLY BM25 call in __init__ for the no-cache version
        self._build_bm25_from_store()

        space = (self.vector_store.collection.metadata or {}).get("hnsw:space", "l2")
        if space != "cosine":
            print(f"[warn] collection hnsw:space = '{space}', not 'cosine'. "
                f"similarity_score = 1 - distance will be misleading.")


    def _build_bm25_from_store(self):
        data = self.vector_store.collection.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metas = data.get("metadatas") or [{} for _ in texts]

        if not texts:
            print("BM25: collection is empty — BM25 disabled.")
            return

        self.bm25_docs = [Document(page_content=t, metadata=m or {})
                        for t, m in zip(texts, metas)]
        tokenized = [self._tokenize(d.page_content) for d in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25: built from {len(self.bm25_docs)} stored chunks (no embeddings).")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(self, query, top_k=5, score_threshold=0.0, mode="hybrid"):
        mode = mode.lower()

        if mode == "vector":
            return self._retrieve_vector(query, top_k, score_threshold)
        
        if mode == "bm25":
            return self._retrieve_bm25(query, top_k)
        
        return self._retrieve_hybrid(query, top_k)

    def compare(self, query, top_k=5):

        print(f"\n{'=' * 72}\nQUERY: {query}\n{'=' * 72}")

        sections = [
            ("VECTOR (cosine, higher=better)", self._retrieve_vector(query, top_k, 0.0)),
            ("BM25 (true score, higher=better)", self._retrieve_bm25(query, top_k)),
            ("HYBRID (RRF, higher=better)", self._retrieve_hybrid(query, top_k)),
        ]

        for title, results in sections:
            print(f"\n--- {title} ---")

            if not results:
                print("  (no results)")

                continue

            for r in results:
                print(f"  {r['rank']}. {self._format_score(r)} "
                      f"{r['documents'].replace(chr(10), ' ')[:90]}...")
        print(f"\n{'=' * 72}\n")


    def _retrieve_vector(self, query, top_k, score_threshold):

        q_emb = self.embedding_manager.embedding_generator([query])[0]

        result = self.vector_store.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
        )

        out = []

        if result["documents"] and result["documents"][0]:
            for i, (doc_id, meta, doc, dist) in enumerate(zip(
                result["ids"][0],
                result["metadatas"][0],
                result["documents"][0],
                result["distances"][0],
            )):
                
                sim = 1.0 - dist  # cosine distance → similarity

                if sim >= score_threshold:
                    out.append({
                        "id": doc_id,
                        "documents": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity_score": sim,
                        "rank": i + 1,
                        "source": "vector",
                    })

        print(f"[vector] {len(out)} document(s) retrieved")

        return out


    def _retrieve_bm25(self, query, top_k):

        if self.bm25 is None:
            print("[bm25] BM25 unavailable.")

            return []
        
        scores = self.bm25.get_scores(self._tokenize(query))

        order = np.argsort(scores)[::-1][:top_k]

        out = []

        for rank, idx in enumerate(order, start=1):
            d = self.bm25_docs[idx]

            out.append({
                "id": f"bm25_{idx}",
                "documents": d.page_content,
                "metadata": d.metadata,
                "distance": None,
                "similarity_score": None,
                "bm25_score": float(scores[idx]),
                "rank": rank,
                "source": "bm25",
            })

        print(f"[bm25] {len(out)} document(s) retrieved")

        return out


    def _retrieve_hybrid(self, query, top_k):

        if self.bm25 is None:
            print("[hybrid] BM25 unavailable — falling back to vector.")

            return self._retrieve_vector(query, top_k, 0.0)

        vec = self._retrieve_vector(query, self.vector_k, 0.0)
        bm  = self._retrieve_bm25(query, self.bm25_k)

        vec_docs = [Document(page_content=r["documents"], metadata=r["metadata"]) for r in vec]
        bm_docs  = [Document(page_content=r["documents"], metadata=r["metadata"]) for r in bm]

        fused = self._rrf([bm_docs, vec_docs], weights=self.weights)

        out = []

        for i, (doc, score) in enumerate(fused[:top_k], start=1):

            out.append({
                "id": f"hybrid_{i}",
                "documents": doc.page_content,
                "metadata": doc.metadata,
                "distance": None,
                "similarity_score": None,
                "rrf_score": score,
                "rank": i,
                "source": "hybrid",
            })

        print(f"[hybrid] {len(out)} document(s) retrieved")

        return out

    def _rrf(self, ranked_lists, weights):

        keyed = {}

        for w, docs in zip(weights, ranked_lists):


            for rank, doc in enumerate(docs, start=1):
                k = doc.page_content

                if k not in keyed:

                    keyed[k] = {"doc": doc, "score": 0.0}
                keyed[k]["score"] += w / (self.RRF_K + rank)

        fused = sorted(keyed.values(), key=lambda x: x["score"], reverse=True)

        return [(item["doc"], item["score"]) for item in fused]

    @staticmethod
    def _format_score(r):
        if r["source"] == "vector":
            return f"[sim={r['similarity_score']:.4f} dist={r['distance']:.4f}]"
        if r["source"] == "bm25":
            return f"[bm25={r['bm25_score']:.4f}]"
        if r["source"] == "hybrid":
            return f"[rrf={r['rrf_score']:.6f}]"
        return "[score=?]"

def run_rag(folder_path=path, reindex=False, top_k=3):
    # 1) Startup
    try:
        em = EmbeddingManager()
        em.embedding_generator(["health check"])   # early failure if Ollama missing
        vs = VectorStoreManager()
        llm = ChatOllama(model=llm_model, temperature=0.1)
    except Exception as e:
        print(f"Startup failed : {e}")
        print(f"Is Ollama running? Did you run: ollama pull {embedding_model} "
              f" and  ollama pull {llm_model} ?")
        return

    # 2) Indexing
    if reindex:
        vs.client.delete_collection(vs.collection_name)
        vs._initialize_store()

    if vs.collection.count() == 0:
        try:
            docs = get_all_documents(folder_path)
            if not docs:
                print(f"No PDF files found in : {folder_path}")
                return
            chunks = chunk_creator(docs)
            embeddings = em.embedding_generator([c.page_content for c in chunks])
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            vs.add_documents(embeddings, chunks, ids)
        except Exception as e:
            print(f"Indexing failed : {e}")
            return
    else:
        print(f"Using the existing vector store ({vs.collection.count()} chunks). "
              "After adding or changing PDFs, run run_rag(reindex=True) once.")

    # 3) Build retriever (BM25 built once at startup from the store)
    retriever = HybridRetriever(em, vs)

    # 4) Query loop
    while True:
        query = input(
            "\nEnter your query "
            "(or 'compare' / 'mode' / 'exit'): "
        ).strip()

        if not query:
            continue
        if query.lower() == "exit":
            print("Exiting.....")
            break
        if query.lower() == "compare":
            q2 = input("Query to compare: ").strip()
            if q2:
                retriever.compare(q2, top_k=top_k)
            continue

        mode = input(
            "Mode [vector/bm25/hybrid] (default=hybrid): "
        ).strip().lower() or "hybrid"
        if mode not in ("vector", "bm25", "hybrid"):
            print(f"Unknown mode '{mode}', using hybrid.")
            mode = "hybrid"

        results = retriever.retrieve(query, top_k=top_k, mode=mode)

        if not results:
            print("\nAnswer: We found no relevant context for the given query.")
            continue

        context = "\n\n".join(r["documents"] for r in results)

        prompt = (
            "You are a helpful question-answering assistant.\n\n"
            "Answer the user's query using ONLY the information provided in the context below.\n"
            "If the context does not contain enough information, clearly state that you do not\n"
            "know based on the available context. Do not invent facts.\n\n"
            f"Context:\n{context}\n\n"
            f"Query:\n{query}\n"
        )

        try:
            response = llm.invoke(prompt)
            print("\nAnswer:", response.content)
        except Exception as e:
            print(f"LLM call failed : {e}")
            print("\nAnswer: Sorry, I couldn't generate an answer right now.")


if __name__ == "__main__":
    run_rag()