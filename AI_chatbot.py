import os
import shutil
import tempfile
import base64
import re
import sqlite3
import json
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv(override=True)

DB_PATH = "chat_history.db"
IMAGE_NAME = "Screenshot 2026-09-09 191654.png"

st.set_page_config(page_title="FAU Smart Document Study Assistant", page_icon="🎓", layout="wide")

# --------------------------------------------------
# Session ID & Directory Initialization
# --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id
SESSION_CHROMA_PATH = f"./chroma_db/{SESSION_ID}"

# --------------------------------------------------
# Isolated & Self-Healing SQLite Database Functions
# --------------------------------------------------
def init_db():
    """Initialize database and safely add missing session_id column if older table schema exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT
        )
    """)
    
    cursor.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in cursor.fetchall()]
    if "session_id" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN session_id TEXT")
        
    conn.commit()
    conn.close()

init_db()

def load_session_history(session_id):
    """Load messages for the current user session only."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT role, content, sources FROM history WHERE session_id = ? ORDER BY id ASC", 
            (session_id,)
        )
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    messages = []
    if not rows:
        messages.append({
            "role": "assistant",
            "content": "👋 Hi! Upload your PDFs in the sidebar and start asking questions.",
            "sources": []
        })
    else:
        for role, content, sources_json in rows:
            sources = json.loads(sources_json) if sources_json else []
            messages.append({
                "role": role,
                "content": content,
                "sources": sources
            })
    return messages

def save_message_to_db(session_id, role, content, sources=None):
    """Save message under the current session ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sources_json = json.dumps(sources) if sources else "[]"
    cursor.execute(
        "INSERT INTO history (session_id, role, content, sources) VALUES (?, ?, ?, ?)", 
        (session_id, role, content, sources_json)
    )
    conn.commit()
    conn.close()

def clear_session_db(session_id):
    """Delete messages only for the active user session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

# Initialize Chat State per Session
if "messages" not in st.session_state:
    st.session_state.messages = load_session_history(SESSION_ID)

# --------------------------------------------------
# LaTeX Formatter Function
# --------------------------------------------------
def format_latex(text: str) -> str:
    """Convert standard LaTeX delimiters and clean unsupported commands for Streamlit rendering."""
    if not isinstance(text, str):
        return text

    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\!\s*$", "", text)

    return text

# --------------------------------------------------
# Background Watermark Function
# --------------------------------------------------
def set_bg_watermark(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = image_path.split(".")[-1].lower()
        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/{ext};base64,{encoded}");
            background-repeat: no-repeat;
            background-position: right 40px bottom 40px;
            background-size: 260px auto;
            background-attachment: fixed;
        }}
        [data-testid="stChatMessage"] {{
            background-color: rgba(255, 255, 255, 0.92) !important;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

set_bg_watermark(IMAGE_NAME)

# --------------------------------------------------
# Main Header
# --------------------------------------------------
if os.path.exists(IMAGE_NAME):
    st.image(IMAGE_NAME, width=280)

st.title("🎓 FAU Smart Document Study Assistant")

# --------------------------------------------------
# Cached Embeddings Model
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

embeddings = load_embeddings()

# --------------------------------------------------
# Session-Isolated Vector Store Management
# --------------------------------------------------
def process_and_index_pdfs(uploaded_files):
    if os.path.exists(SESSION_CHROMA_PATH):
        try:
            shutil.rmtree(SESSION_CHROMA_PATH)
        except Exception:
            pass

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    
    progress_bar = st.progress(0, text="Processing PDFs...")
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        finally:
            os.remove(tmp_path)
        
        progress_bar.progress((idx + 1) / total_files, text=f"Processed {uploaded_file.name}")

    st.info("Generating embeddings and writing to session storage...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=SESSION_CHROMA_PATH
    )
    progress_bar.empty()
    st.empty()
    return vectorstore

def get_unique_sources():
    if not os.path.exists(SESSION_CHROMA_PATH) or not os.listdir(SESSION_CHROMA_PATH):
        return []
    try:
        vectorstore = Chroma(persist_directory=SESSION_CHROMA_PATH, embedding_function=embeddings)
        get_data = vectorstore.get()
        metadatas = get_data.get("metadatas", [])
        return list(set([m["source"] for m in metadatas if m and "source" in m]))
    except Exception:
        return []

def get_balanced_context(query, selected_files=None, top_k_per_doc=6):
    if not os.path.exists(SESSION_CHROMA_PATH) or not os.listdir(SESSION_CHROMA_PATH):
        return "", []
    
    vectorstore = Chroma(persist_directory=SESSION_CHROMA_PATH, embedding_function=embeddings)
    available_sources = get_unique_sources()
    
    target_sources = [s for s in available_sources if s in selected_files] if selected_files else available_sources

    summary_keywords = ["summarize", "summary", "overview", "all files", "both files", "all pdfs", "key points"]
    is_summary_request = any(kw in query.lower() for kw in summary_keywords)

    retrieved_docs = []

    if is_summary_request and len(target_sources) > 0:
        for source in target_sources:
            file_docs = vectorstore.similarity_search(
                query, 
                k=top_k_per_doc, 
                filter={"source": source}
            )
            retrieved_docs.extend(file_docs)
    else:
        if selected_files and len(selected_files) == 1:
            retrieved_docs = vectorstore.similarity_search(query, k=10, filter={"source": selected_files[0]})
        else:
            retrieved_docs = vectorstore.similarity_search(query, k=10)
            if selected_files:
                retrieved_docs = [d for d in retrieved_docs if d.metadata.get("source") in selected_files]

    formatted_context = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Unknown File")
        pg = doc.metadata.get("page", "N/A")
        formatted_context.append(f"--- Document: {src} (Page {pg}) ---\n{doc.page_content}")

    return "\n\n".join(formatted_context), retrieved_docs

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    if os.path.exists(IMAGE_NAME):
        st.image(IMAGE_NAME, width="stretch")
        st.divider()

    st.header("⚙️ Document Management")
    uploaded_files = st.file_uploader(
        "Upload Study Materials (PDFs)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("⚡ Index Documents"):
        if uploaded_files:
            with st.spinner("Indexing session documents..."):
                process_and_index_pdfs(uploaded_files)
                # Reset focus selection in session state after indexing new files
                st.session_state["selected_docs_filter"] = get_unique_sources()
                st.success(f"Indexed {len(uploaded_files)} PDF(s) successfully!")
        else:
            st.warning("Please select at least one PDF file.")

    st.divider()

    # Dynamic File Selection Filter with Persistent Session State
    available_docs = get_unique_sources()
    selected_docs = []
    
    if available_docs:
        st.header("🎯 Focus Search")
        
        # Initialize selected options in session state if not set
        if "selected_docs_filter" not in st.session_state:
            st.session_state.selected_docs_filter = available_docs
        else:
            # Filter out any files that are no longer available
            st.session_state.selected_docs_filter = [
                doc for doc in st.session_state.selected_docs_filter if doc in available_docs
            ]

        selected_docs = st.multiselect(
            "Filter queries to specific file(s):",
            options=available_docs,
            key="selected_docs_filter"
        )
        st.divider()

    if st.button("🗑️ Clear Chat"):
        clear_session_db(SESSION_ID)
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! Upload your PDFs in the sidebar and start asking questions.", "sources": []}
        ]
        st.rerun()

# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

# Render existing session chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_latex(message["content"]))
        if message.get("sources"):
            with st.expander("📚 View Document Sources & Citation Snippets"):
                for idx, src in enumerate(message["sources"], start=1):
                    src_file = src.get("source", "Unknown File")
                    page_num = src.get("page", "N/A")
                    st.markdown(f"**[{idx}] {src_file} — Page {page_num}**")
                    st.caption(src.get("content", ""))

# Capture user input
prompt = st.chat_input("Ask a question about your study materials...")

if prompt:
    user_msg = {"role": "user", "content": prompt, "sources": []}
    st.session_state.messages.append(user_msg)
    save_message_to_db(SESSION_ID, "user", prompt)
    st.rerun()

# Process response if user query was added
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        if not os.path.exists(SESSION_CHROMA_PATH) or not os.listdir(SESSION_CHROMA_PATH):
            st.error("No indexed documents found for your session. Upload and index your PDFs in the sidebar first.")
        else:
            try:
                openrouter_api_key = (
                    os.environ.get("OPENROUTER_API_KEY") 
                    or os.environ.get("OPENAI_API_KEY")
                )

                if not openrouter_api_key:
                    st.error("Missing API Key. Please add OPENAI_API_KEY or OPENROUTER_API_KEY to your environment.")
                    st.stop()

                llm = ChatOpenAI(
                    api_key=openrouter_api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_name="openrouter/free",
                    max_tokens=2000,
                    streaming=True,
                    max_retries=3
                )

                recent_messages = st.session_state.messages[-5:-1]
                chat_history = []
                for msg in recent_messages:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user prompt which might reference previous topics, "
                    "formulate a standalone search query that can be used to search the database. "
                    "Maintain the same language as the user input or the referenced documents. "
                    "Do NOT answer the question, just reformulate it to be self-contained."
                )
                
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                
                query_rewriter = contextualize_q_prompt | llm | StrOutputParser()

                if chat_history:
                    try:
                        standalone_query = query_rewriter.invoke({
                            "chat_history": chat_history,
                            "input": user_prompt
                        }).strip()
                        if not standalone_query:
                            standalone_query = user_prompt
                    except Exception:
                        standalone_query = user_prompt
                else:
                    standalone_query = user_prompt

                context_text, retrieved_docs = get_balanced_context(standalone_query, selected_files=selected_docs, top_k_per_doc=6)

                template = """You are an expert academic study assistant. 

Language Requirement:
- Analyze the primary language used in the provided context or the user's question.
- You MUST write your entire response in that SAME language.
  (For example, if the context or query is in German, answer in German; if Spanish, answer in Spanish; if French, answer in French; etc.)

Answer the user's question directly, comprehensively, and thoroughly based strictly on the provided context.

If the user asks for a SUMMARY (e.g., "summarize", "overview", "key points", "summarize all files", "summarize both files"):
1. You MUST organize your summary into distinct sections for EVERY document present in the context using bold Markdown headings (e.g., `### Document: <filename>`).
2. Provide a detailed summary covering core concepts, theoretical explanations, and conclusions for EACH file separately in the language of the document.
3. Do NOT skip any document found in the context.

When presenting mathematical equations, formulas, variables, or expressions found in the text:
1. Preserve the original mathematical format precisely as written in the source context.
2. Format inline math using single dollar signs (e.g., $E = mc^2$ or $x_i$).
3. Format block or standalone equations using double dollar signs (e.g., $$f(x) = \\int_{{-\\infty}}^{{\\infty}} e^{{-x^2}} dx$$).

Context:
{context}

Question: {question}
"""
                rag_prompt = ChatPromptTemplate.from_template(template)
                rag_chain = rag_prompt | llm | StrOutputParser()

                def generate_stream():
                    for chunk in rag_chain.stream({"context": context_text, "question": user_prompt}):
                        yield chunk

                full_response = st.write_stream(generate_stream)
                formatted_response = format_latex(full_response)

                sources_payload = []
                if retrieved_docs:
                    for doc in retrieved_docs:
                        sources_payload.append({
                            "source": doc.metadata.get("source", "Unknown File"),
                            "page": doc.metadata.get("page", "N/A"),
                            "content": doc.page_content
                        })

                    with st.expander("📚 View Document Sources & Citation Snippets"):
                        for idx, src in enumerate(sources_payload, start=1):
                            st.markdown(f"**[{idx}] {src['source']} — Page {src['page']}**")
                            st.caption(src['content'])

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": formatted_response,
                    "sources": sources_payload
                })
                save_message_to_db(SESSION_ID, "assistant", formatted_response, sources_payload)

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
