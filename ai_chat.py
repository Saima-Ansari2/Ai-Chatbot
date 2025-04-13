import streamlit as st
import os
import tempfile
import pdfplumber
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set page config
st.set_page_config(
    page_title="RAG Chatbot with Gemini",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Utility Functions
def setup_api_key(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

def upload_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except:
        return None

def parse_pdf(content):
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except:
        return None

def create_document_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def init_embedding_model():
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except:
        return None

def store_embeddings(model, texts, persist_directory):
    try:
        return Chroma.from_texts(
            texts=texts,
            embedding=model,
            persist_directory=persist_directory
        )
    except:
        return None

def query_with_full_context(query, vectorstore, k=3, temperature=0.2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=temperature
        ),
        retriever=retriever,
        memory=memory
    )
    result = qa_chain({"question": query})
    return result["answer"], "", []

# Document Processing
def process_documents(uploaded_files):
    if st.session_state.embedding_model is None:
        st.session_state.embedding_model = init_embedding_model()
        if st.session_state.embedding_model is None:
            st.sidebar.error("Failed to initialize embedding model.")
            return

    all_chunks = []
    processed_file_names = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        pdf_content = upload_pdf(pdf_path)
        if not pdf_content:
            continue

        text = parse_pdf(pdf_content)
        if not text:
            continue

        chunks = create_document_chunks(text)
        all_chunks.extend(chunks)
        processed_file_names.append(uploaded_file.name)

        os.unlink(pdf_path)

    if all_chunks:
        st.session_state.vectorstore = store_embeddings(
            st.session_state.embedding_model,
            all_chunks,
            persist_directory="./streamlit_chroma_db"
        )
        st.session_state.processed_files = processed_file_names
        st.sidebar.success("Documents processed successfully!")

# Main Chatbot Interface
def main():
    st.sidebar.title("RAG Chatbot")
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if api_key and st.sidebar.button("Set API Key"):
        setup_api_key(api_key)
        st.success("API Key set!")

    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.sidebar.button("Process Documents"):
        process_documents(uploaded_files)

    st.sidebar.slider("Chunks (k)", 1, 10, 3, key="k_value")
    st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="temperature")

    st.title("📚 RAG Chatbot with Gemini")

    if st.session_state.vectorstore:
        display_chat()
        user_query = st.chat_input("Ask your question...")
        if user_query:
            handle_user_query(user_query)
    else:
        st.info("Upload and process documents to start chatting.")

def handle_user_query(query):
    st.session_state.conversation.append({"role": "user", "content": query})
    thinking = st.empty()
    thinking.info("🤔 Thinking...")

    try:
        response, context, _ = query_with_full_context(
            query,
            st.session_state.vectorstore,
            k=st.session_state.k_value,
            temperature=st.session_state.temperature
        )
        st.session_state.conversation.append({"role": "assistant", "content": response, "context": context})
        thinking.empty()
        display_chat()
    except Exception as e:
        thinking.empty()
        st.error(f"Error: {e}")

def display_chat():
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("context"):
                with st.expander("View source context"):
                    st.text(msg["context"])

if __name__ == "__main__":
    main()
