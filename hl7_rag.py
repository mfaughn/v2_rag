import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
import pickle

class HL7RAGPipeline:
    def __init__(self, docs_dir: str, db_dir: str):
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def load_documents(self):
        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()

    def process_documents(self):
        # Parent splitter for larger chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
    
        # Child splitter for more granular retrieval
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
    
        # Create storage directory if it doesn't exist
        os.makedirs(self.db_dir, exist_ok=True)
    
        # Initialize store with custom document serialization
        class SerializableStore(LocalFileStore):
            def mset(self, key_value_pairs):
                serialized_pairs = [
                    (k, pickle.dumps(v)) for k, v in key_value_pairs
                ]
                super().mset(serialized_pairs)
            
            def mget(self, keys):
                serialized_values = super().mget(keys)
                return [
                    pickle.loads(v) if v is not None else None 
                    for v in serialized_values
                ]
    
        store = SerializableStore(self.db_dir)
    
        # Initialize Chroma with persistence configuration
        vectorstore = Chroma(
            collection_name="hl7v291",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(self.db_dir, "chroma"),
            collection_metadata={"hnsw:space": "cosine"}
        )
    
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
    
        # Load and add documents
        docs = self.load_documents()
        retriever.add_documents(docs)
    
        return retriever
        
    def query(self, retriever, query: str, k: int = 4):
        return retriever.invoke(query)