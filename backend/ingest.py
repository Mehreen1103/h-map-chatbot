import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PDF_PATH = "data/pdfs/family_database_ovg_vi_dec_2024.pdf"
VECTORSTORE_PATH = "vectorstore/faiss_index"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Embeddings
# ✅ FREE & LOCAL embeddings (NO API, NO LIMITS)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)

# Save locally
os.makedirs(VECTORSTORE_PATH, exist_ok=True)
vectorstore.save_local(VECTORSTORE_PATH)

print("✅ Vectorstore created successfully!")
