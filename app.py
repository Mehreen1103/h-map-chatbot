import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------
# ENV
# ---------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VECTORSTORE_PATH = "vectorstore/faiss_index"

# ---------------------------------
# Streamlit setup (MUST be first UI call)
# ---------------------------------
st.set_page_config(page_title="HabiMate", layout="wide")
st.markdown('<h1 style="text-align: center;">HabiMate</h1>', unsafe_allow_html=True)

# ---------------------------------
# Load Vectorstore
# ---------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------------------------------
# Gemini LLM
# ---------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)

# ---------------------------------
# Prompt
# ---------------------------------
prompt = ChatPromptTemplate.from_template("""
You are a friendly assistant for GreenLead.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
""")

# ---------------------------------
# RAG Chain
# ---------------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# ---------------------------------
# Chat state
# ---------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("HabiMate", "Hi there! I’m HabiMate, your AI-powered guide for informed and data-driven decisions that support safer homes and resilient communities. How can I help today?")
    ]

# ---------------------------------
# Chat UI
# ---------------------------------
# ---------------------------------
# Chat UI
# ---------------------------------
chat_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

#chat-box {
    height: 250px;                /* slightly smaller than before */
    overflow-y: auto; 
    padding: 8px;
    background: white; 
    border-radius: 8px;
    font-family: 'Poppins', sans-serif;
    font-size: 14px;
    margin-bottom: 10px;          /* spacing before input box */
    border: 2px solid #4CAF50;
    box-shadow: 0 0 20px rgba(76, 175, 80, 0.5), 0 0 40px rgba(76, 175, 80, 0.2);
}

.message {
    max-width: 70%; 
    padding: 6px 10px;            /* reduced padding for tighter look */
    margin-bottom: 5px;           /* tighter spacing between messages */
    border-radius: 12px;
    line-height: 1.3;             /* tighter line spacing */
    word-wrap: break-word;
}

.user {
    background: #e8f0fe; 
    margin-left: auto;
}

.bot {
    background: #d7f2d8;
    margin-right: auto;
}
</style>
<div id="chat-box">
"""

# Add chat messages
for speaker, msg in st.session_state.chat_history:
    cls = "user" if speaker == "You" else "bot"
    chat_html += f"<div class='message {cls}'><b>{speaker}:</b> {msg}</div>"

chat_html += "</div>"

# Display the chat box
components.html(chat_html, height=300)   # make the iframe height smaller


# ---------------------------------
# Input
# ---------------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question")
    submit = st.form_submit_button("Send")

if submit and user_input:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response.content))
        st.rerun()
