import os
import tempfile
import base64
import re
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

# Load environment variables from .env file (if running locally)
load_dotenv(override=True)

CHROMA_PATH = "./chroma_db"
IMAGE_NAME = "Screenshot 2026-09-09 191654.png"

st.set_page_config(page_title="FAU Smart Document Study Assistant", page_icon="🎓", layout="wide")

# --------------------------------------------------
# LaTeX Formatter Function
# --------------------------------------------------
def format_latex(text: str) -> str:
    """Convert standard LaTeX delimiters and clean unsupported commands for Streamlit rendering."""
    if not isinstance(text, str):
        return text

    # Standardize LaTeX delimiters to Streamlit-compatible Markdown
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")

    # Remove problematic LaTeX wrapper commands often produced by LLMs
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
# Vector Store Management
# --------------------------------------------------
def process_and_index_pdfs(uploaded_files):
    # Reset Chroma collection safely via API to prevent Windows permission locks
    try:
        existing_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        existing_store.delete_collection()
    except Exception:
        pass  # Collection does not exist yet

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
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        finally:
            os.remove(tmp_path)
        
        progress_bar.progress((idx + 1) / total_files, text=f"Processed {uploaded_file.name}")

    st.info("Generating embeddings and writing to disk...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    progress_bar.empty()
    st.empty()
    return vectorstore

def get_retriever():
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    return None

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
            with st.spinner("Indexing all documents..."):
                process_and_index_pdfs(uploaded_files)
                st.success(f"Indexed {len(uploaded_files)} PDF(s) successfully!")
        else:
            st.warning("Please select at least one PDF file.")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! Upload your PDFs in the sidebar and start asking questions."}
        ]
        st.rerun()

# --------------------------------------------------
# Chat Interface
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! Upload your PDFs in the sidebar and start asking questions."}
    ]

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_latex(message["content"]))

# Capture input
prompt = st.chat_input("Ask a question about your study materials...")

if prompt:
    # Append user prompt and trigger immediate rerun to keep history state rendering synced
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Execute model logic if the last element in session state is an unanswered user query
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("Searching across documents..."):
            retriever = get_retriever()
            
            if retriever is None:
                st.error("No indexed documents found. Upload and index your PDFs in the sidebar first.")
            else:
                try:
                    openrouter_api_key = (
                        os.environ.get("OPENROUTER_API_KEY") 
                        or os.environ.get("OPENAI_API_KEY")
                    )

                    if not openrouter_api_key:
                        st.error("Missing API Key. Please add OPENAI_API_KEY or OPENROUTER_API_KEY to your environment/secrets.")
                        st.stop()

                    groq_chat = ChatOpenAI(
                        api_key=openrouter_api_key,
                        openai_api_base="https://openrouter.ai/api/v1",
                        model_name="openrouter/free",
                        max_tokens=1024,
                        max_retries=3
                    )

                    # Build history payload
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))

                    # Contextualize query
                    contextualize_q_system_prompt = (
                        "Given a chat history and the latest user prompt which might reference previous topic, "
                        "formulate a standalone search query that can be used to search the database. "
                        "Do NOT answer the question, just reformulate it to be self-contained."
                    )
                    
                    contextualize_q_prompt = ChatPromptTemplate.from_messages([
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ])
                    
                    query_rewriter = contextualize_q_prompt | groq_chat | StrOutputParser()

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

                    # Document retrieval
                    retrieved_docs = retriever.invoke(standalone_query)
                    
                    def format_docs(docs):
                        return "\n\n--- Document Chunk ---\n\n".join(doc.page_content for doc in docs)

                    context_text = format_docs(retrieved_docs)

                    # RAG Answer Chain
                    template = """Answer the user's question directly and thoroughly based ONLY on the provided context.

When presenting mathematical equations, formulas, variables, or expressions found in the text:
1. Preserve the original mathematical format precisely as written in the source context.
2. Format inline math using single dollar signs (e.g., $E = mc^2$ or $x_i$).
3. Format block or standalone equations using double dollar signs (e.g., $$f(x) = \\int_{{-\\infty}}^{{\\infty}} e^{{-x^2}} dx$$).

Context:
{context}

Question: {question}
"""
                    rag_prompt = ChatPromptTemplate.from_template(template)
                    rag_chain = rag_prompt | groq_chat | StrOutputParser()
                    
                    raw_response = rag_chain.invoke({
                        "context": context_text,
                        "question": user_prompt
                    })

                    if not raw_response or not raw_response.strip():
                        raw_response = "I couldn't retrieve a specific answer from the document context. Please try asking your question with more detail."

                    clean_response = format_latex(raw_response)
                    
                    # Store response and update view state
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")