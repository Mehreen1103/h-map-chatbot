# import os
# from dotenv import load_dotenv

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# from langchain_community.chains import RetrievalQA

# # Load environment variables
# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Paths
# VECTORSTORE_PATH = "vectorstore"

# # Load embeddings (same one used during ingest)
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # Load FAISS index
# db = FAISS.load_local(
#     VECTORSTORE_PATH,
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# # Gemini LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     google_api_key=GEMINI_API_KEY,
#     temperature=0.3
# )

# # Retrieval QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={"k": 4}),
#     return_source_documents=True
# )

# def ask_question(query: str):
#     """
#     Ask a question to the FAISS + Gemini system
#     """
#     response = qa_chain({"query": query})
#     return response["result"]
