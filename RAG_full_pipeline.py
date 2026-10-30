from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
import hashlib

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
            metadata={"description": "This is demo RAG"}
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


class RAGretriever:
    def __init__(self,embedding_manager,vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self,query,top_k=5,score_threshold=0.0):

        query_embedding = self.embedding_manager.embedding_generator([query])[0]

        result = self.vector_store.collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k
        )

        retrieved_documents = []

        if result["documents"] and result["documents"][0]:
            ids = result["ids"][0]
            metadatas = result["metadatas"][0]
            documents = result["documents"][0]
            distances = result["distances"][0]

            for i, (doc_id,metadata,document,distance) in enumerate(zip(ids,metadatas,documents,distances)):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_documents.append({
                        "id" : doc_id,
                        "documents" : document,
                        "metadata" : metadata,
                        "distance" : distance,
                        "similarity_score" : similarity_score,
                        "rank" : i + 1
                    })

            print(f"{len(retrieved_documents)} Documents Retrieved")

        else:
            print("No Documents Found")

        return retrieved_documents


def run_rag(folder_path=path, reindex=False, top_k=3):
    """Runs the whole RAG system: set up -> load PDFs -> chunk -> embed -> store -> retrieve -> answer."""

    # 1) Set up the embedding model, vector store and LLM
    try:
        em = EmbeddingManager()
        em.embedding_generator(["health check"])    # fails early if Ollama / the model is missing
        vs = VectorStoreManager()
        llm = ChatOllama(model=llm_model, temperature=0.1)
    except Exception as e:
        print(f"Startup failed : {e}")
        print(f"Is Ollama running? Did you run: ollama pull {embedding_model}  and  ollama pull {llm_model} ?")
        return

    # 2) Index the PDFs (only when the database is empty, or when reindex=True)
    if reindex:
        vs.client.delete_collection(vs.collection_name)  # wipe the old data
        vs._initialize_store()          # create a fresh empty collection

    if vs.collection.count() == 0:
        try:
            docs = get_all_documents(folder_path)           # load all PDFs
            if not docs:
                print(f"No PDF files found in : {folder_path}")
                return

            chunks = chunk_creator(docs)                    # split into chunks
            embeddings = em.embedding_generator([c.page_content for c in chunks])# embed
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            vs.add_documents(embeddings, chunks, ids)       # store
        except Exception as e:
            print(f"Indexing failed : {e}")
            return
    else:
        print(f"Using the existing vector store ({vs.collection.count()} chunks). "
              "After adding or changing PDFs, run run_rag(reindex=True) once.")

    # 3) Ask questions: retrieve -> build the prompt -> ask the LLM
    retriever = RAGretriever(em, vs)

    while True:
        query = input("\nEnter your query (or 'exit' to quit): ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Exiting.....")
            break

        # retrieve the most relevant chunks and join them into one context block
        results = retriever.retrieve(query, top_k)
        context = "\n\n".join(doc["documents"] for doc in results)

        if not context.strip():
            print("\nAnswer: We found no relevant context for the given query.")
            continue

        # the prompt that is sent to the LLM
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