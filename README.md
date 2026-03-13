## 📌 Project Overview

# HabiMate 🌿

An AI-powered RAG (Retrieval-Augmented Generation) chatbot built with Streamlit and LangChain, designed to help users make informed, data-driven decisions for safer homes and resilient communities.

---

## Overview

HabiMate uses a FAISS vector store to retrieve relevant context from local PDF documents and passes it to **Google Gemini (gemini-2.5-flash-lite)** to generate accurate, grounded answers. It features a clean chat UI built entirely in Streamlit.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| LLM | Google Gemini 2.5 Flash Lite (via LangChain) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Backend (API) | FastAPI |
| Data Source | PDF documents |

---

## Project Structure
```
├── .devcontainer/          # Dev container config
├── backend/                # FastAPI backend
├── data/
│   └── pdfs/
│       └── family_database_ovg_vi_dec_2024.pdf
├── vectorstore/            # FAISS index (generated)
├── app.py                  # Main Streamlit app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How It Works

1. PDF documents are loaded from `data/pdfs/` and chunked into segments
2. Each chunk is embedded using `all-MiniLM-L6-v2` and stored in a FAISS index
3. When a user asks a question, the top 5 most relevant chunks are retrieved
4. The retrieved context + question are passed to Gemini via a LangChain RAG chain
5. The answer is displayed in the chat UI

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/habimate.git
cd habimate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Build the vector store

Make sure your PDFs are in `data/pdfs/`, then run your ingestion script to generate the FAISS index in `vectorstore/faiss_index`.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Backend API (FastAPI)

A lightweight FastAPI backend is also available for programmatic access:
```bash
uvicorn backend.main:app --reload
```

**Endpoint:**
```
GET /search?query=your+question+here
```

Returns:
```json
{
  "result": "AI-generated answer based on retrieved context"
}
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API key |

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

- `streamlit`
- `langchain`
- `langchain-google-genai`
- `langchain-community`
- `faiss-cpu`
- `sentence-transformers`
- `python-dotenv`
- `fastapi`
- `uvicorn`

---

## Notes

- The chatbot answers **only from the provided PDF context** — it will not hallucinate answers outside the retrieved documents.
- The FAISS index must be built before running the app. The `vectorstore/` folder is not committed to the repo (see `.gitignore`).
![Chatbot](https://github.com/user-attachments/assets/a7e33dc0-f1c3-410b-b7e0-71dd880a0f4c)
