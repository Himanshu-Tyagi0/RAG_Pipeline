from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
import hashlib

def get_all_documents(folder_path="C:\\Users\\tyagi\\langchain_projects\\RAG_Understanding"):
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


def file_hash(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


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
            model="qwen3-embedding"
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

    def hash_exists(self, fhash):
        res = self.collection.get(where={"file_hash": fhash}, limit=1, include=[])
        return len(res["ids"]) > 0

    def delete_by_source(self, filename):
        # removes old chunks if a file with this name was ingested before (e.g. it was edited)
        self.collection.delete(where={"source_file": filename})

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


def generate_output(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k)

    # Extract retrieved document content
    context = "\n\n".join(
        doc["documents"] for doc in results
    ) if results else ""

    # Handle empty retrieval
    if not context.strip():
        return "We found no relevant context for the given query."

    # Insert the actual context and query
    prompt = f"""
                You are a helpful question-answering assistant.

                Answer the user's query using ONLY the information
                provided in the context below.

                If the context does not contain enough information,
                clearly state that you do not know based on the
                available context. Do not invent facts.

                Context:
                {context}

                Query:
                {query}

    """


    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"LLM call failed : {e}")
        print("Sorry, i couldn't generate an answer right now.")



def ingest_folder(folder_path, embedding_manager, vector_store):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not Found : {folder_path}")

    added, skipped = 0, 0

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, filename)

        try:
            fhash = file_hash(pdf_path)

            # Same content already in the vector store -> skip (no load/chunk/embed)
            if vector_store.hash_exists(fhash):
                print(f"Skipping (already indexed): {filename}")
                skipped += 1
                continue

            # Same filename but changed content -> remove the old chunks first
            vector_store.delete_by_source(filename)

            docs = PyPDFLoader(pdf_path).load()
            chunks = chunk_creator(docs)

            for c in chunks:
                c.metadata["file_hash"] = fhash
                c.metadata["source_file"] = filename

            embeddings = embedding_manager.embedding_generator(
                [c.page_content for c in chunks]
            )
            ids = [f"{fhash}_{i}" for i in range(len(chunks))]

            vector_store.add_documents(embeddings, chunks, ids)
            added += 1
            print(f"Indexed: {filename} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"Failed on {filename} : {e}")

    print(f"Ingestion done. New: {added}, Skipped: {skipped}")


em = EmbeddingManager()
vs = VectorStoreManager()

ingest_folder("C:\\Users\\tyagi\\langchain_projects\\RAG_Understanding", em, vs)

retriever = RAGretriever(em, vs)
llm = ChatOllama(model="llama2", temperature=0.1)

while True:
    query = input("\nEnter your query (or 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting.....")
        break

    answer = generate_output(query, retriever, llm)
    print("\nAnswer:", answer)