from anthropic import Anthropic
from typing import List
from langchain.schema import Document

def query_claude(retriever, anthropic_client: Anthropic, query: str):
    # Get relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Construct context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Construct prompt
    prompt = f"""Context from HL7 v2.9.1 documentation:
{context}

Based on the above context, please answer the following question:
{query}"""
    
    # Query Claude
    response = anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    return response.content