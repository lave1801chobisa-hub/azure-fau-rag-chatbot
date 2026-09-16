
import os
import tempfile
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv(override=True)

CHROMA_PATH = "./chroma_db"
st.set_page_config(page_title="FAU Smart Document Study Assistant", page_icon="🎓", layout="wide")

IMAGE_NAME = "Screenshot 2026-09-09 191654.png"

# Helper function to fix math formatting for Streamlit
def format_latex(text: str) -> str:
    """Converts standard LaTeX delimiters to Streamlit-compatible dollar signs."""
    if not isinstance(text, str):
        return text
    # Replace inline math delimiters
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    # Replace block math delimiters
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

# --------------------------------------------------
# Background Watermark
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
        st.image(IMAGE_NAME, use_container_width=True)
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
            {"role": "assistant", "content": "👋 Hi! Ask me anything about your study materials."}
        ]
        st.rerun()

# --------------------------------------------------
# Chat Interface
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! Upload your PDFs in the sidebar and start asking questions."}
    ]

# Render chat history with LaTeX formatting applied
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_latex(message["content"]))

prompt = st.chat_input("Ask a question about your study materials...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching across documents..."):
            retriever = get_retriever()
            
            if retriever is None:
                st.error("No indexed documents found. Upload and index your PDFs in the sidebar first.")
            else:
                try:
                    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
                    groq_chat = ChatOpenAI(
                        api_key=openrouter_api_key,
                        openai_api_base="https://openrouter.ai/api/v1",
                        model_name="openrouter/free",
                        max_tokens=1024
                    )

                    # Explicitly instruction in prompt to format math with $
                    template = """Answer the question directly and thoroughly based ONLY on the provided context.
When writing mathematical formulas or equations, always wrap inline formulas with single dollar signs (e.g. $E=mc^2$) and display formulas with double dollar signs (e.g. $$E=mc^2$$).

Context:
{context}

Question: {question}
"""
                    rag_prompt = ChatPromptTemplate.from_template(template)

                    def format_docs(docs):
                        return "\n\n--- Document Chunk ---\n\n".join(doc.page_content for doc in docs)

                    chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | rag_prompt
                        | groq_chat
                        | StrOutputParser()
                    )

                    raw_response = chain.invoke(prompt)
                    
                    
                    # Clean and format math formula tags
                    clean_response = format_latex(raw_response)
                    
                    st.markdown(clean_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})

                except Exception as e:
                    st.error(f"Error: {str(e)}")

               
import re


def format_latex(text: str) -> str:
    """Convert standard LaTeX delimiters and clean unsupported commands for Streamlit."""
    
    if not isinstance(text, str):
        return text

    # Replace inline math delimiters: \( ... \) -> $ ... $
    text = text.replace(r"\(", "$").replace(r"\)", "$")

    # Clean malformed/extra block-math delimiters
    text = text.replace("\n$$", "\n$$")

    # Remove unsupported/problematic LaTeX wrapper commands such as \boxed{...}
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)

    # Remove trailing standalone \! often appearing next to math blocks
    text = re.sub(r"\\!\s*$", "", text)

    return text