# main.py
from hl7_rag import HL7RAGPipeline

def main():
    pipeline = HL7RAGPipeline(
        docs_dir="./hl7_docs",
        db_dir="./hl7_db"
    )
    
    # Process documents and create retriever
    retriever = pipeline.process_documents()
    
    # Example query
    results = pipeline.query(
        retriever,
        "What are the specifications for the XAD data type?"
    )
    
    for doc in results:
        print(f"Content: {doc.page_content}\n")
        print(f"Source: {doc.metadata}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()