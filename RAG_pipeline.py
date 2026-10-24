import os
import chromadb
import uuid
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_all_pdf(folder_path=None):
    if folder_path is None:
        folder_path = os.environ.get("DATA_DIR",os.getcwd(),"DATA")

    num_docs = 0
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(folder_path,filename)

            loader = PyPDFLoader(pdf_path)
            doc = loader.load()


            all_docs.extend(doc)
            num_docs +=1

    if len(all_docs) == 0:
        raise FileNotFoundError(f"No, No file not found  in : {pdf_path}. Please add your documents in the given folder path")

    print("Number of documents added :",num_docs)
    print("Number of pages :",len(all_docs))
    return all_docs


all_pdf = load_all_pdf("C:\\Users\\tyagi\\langchain_projects\\RAG_Understanding")


def chunk_maker(documents,chunk_size=1000,chunk_overlap=100):
    chunked = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = chunked.split_documents(documents=documents)

    return all_chunks

chunks = chunk_maker(all_pdf)

class EmbeddingManager:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name= "sentence-transformers/all-MiniLM-L6-v2",
        )

        print("Model is loaded")

    def generate_embeddings(self,text):

        embeddings = self.model.embed_documents(texts=text)

        print("EMbedding dimensions :",len(embeddings[0]))

        return embeddings


embedding_manager = EmbeddingManager()

class VectorStoreManager:
    def __init__(self,persis_directory = None,collection_name = "pdf_documents"):
        if persis_directory is None:
            persis_directory = os.environ.get("VECTOR_STORE_DIR",os.path.join(os.getcwd(),"DATA","vector_store"))

        self.persis_directory = persis_directory
        self.collection_name = collection_name

        self.collection = None
        self.client = None

        self._initialize_store()

    def _initialize_store(self):

        os.makedirs(self.persis_directory,exist_ok=True)

        self.client = chromadb.PersistentClient(self.persis_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata="THis is just for my practice"
        )

        print("Vector store initialize with the name of :",self.collection_name)
        print("DOcuments in collection :",self.collection.count())

    def add_documents(self,documents,emebddings):
        if len(documents) != len(emebddings):
            print("The length of documents does not match with length of embeddings")

        ids = []
        metadatas = []
        document_content = []
        emebdding_list = []

        for i,(doc,emebdding) in enumerate(zip(documents,emebddings)):

            doc_id = F"doc_{uuid.uuid4}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["index"] =  i+ 1
            metadata["length"] = len(doc.page_content)
            metadatas.append(metadata)

            document_content.append(doc.page_content)

            emebdding_list.append(emebdding)

            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_content,
                embeddings=emebdding_list
            )

        print("Total documents added :",len(document_content))
        print("Documents in collection :",self.collection.count())


vectore_store = VectorStoreManager()

text = [doc.page_content for doc in chunks]

embeddings = embedding_manager.generate_embeddings(text=text)

vectore_store.add_documents(documents=chunks,emebddings=embeddings)

class RAGRetriever:
    def __init__(self,embedding_manager,vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self,query,top_k=4,score_threshold=0):
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k
        )

        retrieved_docs = []

        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadata"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i,(id,metadata,document,distance) in enumerate(zip(ids,metadatas,documents,distances)):

                similarity_socre = 1 - distance

                if similarity_socre >= score_threshold:
                    retrieved_docs.append(
                        {
                            "id" : id,
                            "metadata" : metadata,
                            "document" : document,
                            "distance" : distance,
                            "similarity_score" : similarity_socre,
                            "rank" : i+1
                        }
                    )
            print("Retrieved documents length :",len(retrieved_docs))

        else:
            print("No documents found")

        return retrieved_docs

