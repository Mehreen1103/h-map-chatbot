from fastapi import FastAPI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.chains import RetrievalQA  
from langchain_community.llms import HuggingFacePipeline
app = FastAPI()

# Load embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings)

# Setup LLM (local, free, no API key)
hf_pipeline = pipeline(
    "text-generation",
    model="gemini-1.5-flash",
    temperature=0.7,
    max_new_tokens=200
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

@app.get("/search")
def search(query: str):
    result = qa_chain.run(query)
    return {"result": result}