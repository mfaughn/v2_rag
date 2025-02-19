The initial work here is based on asking Claude Sonnet 3.5 how to create a RAG pipeline.

Claude didn't get it right on the first go and there has been some back and forth to get things right.  Below is my best effort at consolidating things into what _might_ get things to work on the first go.

After cloning this project you will need to execute the following in the root directory:

```
python -m venv venv
source venv/bin/activate
pip install langchain-community langchain-huggingface langchain-chroma pypdf unstructured transformers torch sentence-transformers
mkdir -p hl7_db
mkdir -p hl7_docs
```


Below are the steps suggested by Claude.  I haven't gotten past step 2 yet.

1. Place your HL7 v2.9.1 PDFs in the hl7_docs directory
2. Run the processing script (main.py) to create the vector store
3. Test queries to verify retrieval quality
4. Fine-tune chunking parameters if needed
5. Integrate with Claude as shown above
