# RAG Agent – Upload Document (PDF, Word, Excel, PowerPoint, Python, Jupyter Notebook) and ask questions about it.
# Stack: OpenAI API + LangChain + FAISS + Streamlit

import os
import json
import tempfile
import docx
import pandas as pd
import streamlit as st
from pptx import Presentation

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Document RAG Agent", page_icon="📄", layout="wide")

# -----------------------------------------------------------------------------
# Global styling
# -----------------------------------------------------------------------------
# Color palette, backgrounds, and button/input colors now come from the native
# Streamlit theme in .streamlit/config.toml (primaryColor, backgroundColor,
# secondaryBackgroundColor, textColor, font) instead of being hardcoded here.
# This CSS block is kept minimal — only for the handful of custom elements
# (pill badge, ghost button, dropzone) that the theme engine doesn't cover.
st.markdown(
    """
    <style>
    /* Hide default streamlit chrome for a cleaner look */
    #MainMenu, footer, header {visibility: hidden;}

    h1 {
        letter-spacing: -0.02em;
    }

    /* Secondary / ghost buttons — theme's primaryColor used for the border/text */
    .ghost-btn button {
        background-color: transparent !important;
        color: var(--primary-color, #4F46E5) !important;
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
    }
    .ghost-btn button:hover {
        border-color: var(--primary-color, #4F46E5) !important;
    }

    /* Cards — Streamlit's native bordered container, spacing only */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1.75rem 2rem;
    }

    /* Badge / pill */
    .pill {
        display: inline-block;
        background-color: rgba(79, 70, 229, 0.12);
        color: var(--primary-color, #4F46E5);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(79, 70, 229, 0.03);
        border: 1.5px dashed rgba(79, 70, 229, 0.35);
        border-radius: 12px;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 12px;
        padding: 0.5rem 0.25rem;
    }

    .muted {
        opacity: 0.65;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state defaults
# -----------------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "login"          # "login" -> "app"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# -----------------------------------------------------------------------------
# Page 1 — API key gate
# -----------------------------------------------------------------------------
def render_login():
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<span class="pill">📄 Document RAG Agent</span>', unsafe_allow_html=True)
            st.markdown("### Welcome")
            st.markdown(
                '<p class="muted">Enter your OpenAI API key to continue. '
                "It's used only for this session and is never stored.</p>",
                unsafe_allow_html=True,
            )

            with st.form("api_key_form", clear_on_submit=False):
                key_input = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    placeholder="sk-...",
                )
                submitted = st.form_submit_button("Continue →", use_container_width=True)

            if submitted:
                if not key_input.strip():
                    st.error("Please enter an API key to continue.")
                elif not key_input.strip().startswith(("sk-", "sess-")):
                    st.error("That doesn't look like a valid OpenAI API key. Please check and try again.")
                else:
                    st.session_state.api_key = key_input.strip()
                    os.environ["OPENAI_API_KEY"] = key_input.strip()
                    st.session_state.stage = "app"
                    st.rerun()

            st.markdown(
                '<p class="muted">Don\'t have a key? Create one at '
                '<a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com</a>.</p>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Document loading / indexing
# -----------------------------------------------------------------------------
def build_vectorstore(file_bytes: bytes, file_name: str, size: int, overlap: int) -> FAISS:
    """Load document dynamically by extension, split into chunks, embed, and store in FAISS."""
    ext = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif ext in [".docx", ".doc"]:
            doc = docx.Document(tmp_path)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text])
            docs = [Document(page_content=full_text, metadata={"source": file_name})]
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp_path)
            content = df.to_string(index=False)
            docs = [Document(page_content=content, metadata={"source": file_name})]
        elif ext in [".pptx", ".ppt"]:
            prs = Presentation(tmp_path)
            docs = []
            for i, slide in enumerate(prs.slides, start=1):
                parts = []
                for shape in slide.shapes:
                    if shape.has_text_frame and shape.text_frame.text:
                        parts.append(shape.text_frame.text)
                    if shape.has_table:
                        for row in shape.table.rows:
                            parts.append(" | ".join(cell.text for cell in row.cells))
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text
                    if notes:
                        parts.append(f"[Speaker notes] {notes}")
                slide_text = "\n".join(p for p in parts if p)
                if slide_text:
                    docs.append(
                        Document(
                            page_content=slide_text,
                            metadata={"source": file_name, "page": i},
                        )
                    )
        elif ext == ".ipynb":
            with open(tmp_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            docs = []
            for i, cell in enumerate(nb.get("cells", []), start=1):
                source = cell.get("source", [])
                text = "".join(source) if isinstance(source, list) else source
                if not text.strip():
                    continue
                cell_type = cell.get("cell_type", "code")
                content = f"[{cell_type} cell]\n{text}"
                # Include text-based outputs (e.g. print statements, repr) for code cells
                if cell_type == "code":
                    out_texts = []
                    for out in cell.get("outputs", []):
                        if "text" in out:
                            out_text = out["text"]
                            out_texts.append(
                                "".join(out_text) if isinstance(out_text, list) else out_text
                            )
                        elif "data" in out and "text/plain" in out.get("data", {}):
                            out_text = out["data"]["text/plain"]
                            out_texts.append(
                                "".join(out_text) if isinstance(out_text, list) else out_text
                            )
                    if out_texts:
                        content += "\n[Output]\n" + "\n".join(out_texts)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": file_name, "page": i},
                    )
                )
        elif ext == ".py":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs = [Document(page_content=text, metadata={"source": file_name})]
        else:
            # Plain-text files aren't always UTF-8 (Notepad often saves as
            # UTF-16 or Windows-1252), and TextLoader raises a RuntimeError
            # instead of falling back. Try a few common encodings ourselves.
            with open(tmp_path, "rb") as f:
                raw = f.read()
            text = None
            for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                text = raw.decode("utf-8", errors="ignore")
            docs = [Document(page_content=text, metadata={"source": file_name})]
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------------------------------------------------------
# RAG chain
# -----------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions strictly using the "
        "provided context from an uploaded document.\n"
        "Rules:\n"
        "1. Answer ONLY from the context below.\n"
        "2. If the answer is not in the context, say: "
        "\"I couldn't find that in the document.\"\n"
        "3. Cite source details or page numbers if available.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    """Join retrieved chunks, tagging each with its page/source number."""
    return "\n\n".join(
        f"[Source Info: Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def get_chain(vectorstore: FAISS, model: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ), retriever


# -----------------------------------------------------------------------------
# Page 2 — Main app
# -----------------------------------------------------------------------------
def render_app():
    # ---- Sidebar: settings ----
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown(
            f'<p class="muted">Key: <code>{st.session_state.api_key[:6]}••••••••</code></p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("Change API key", use_container_width=True):
            st.session_state.stage = "login"
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("file_sig", None)
            st.session_state.pop("messages", None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Model & retrieval**")
        model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Retrieved chunks (k)", 2, 10, 4)

    # ---- Header ----
    st.markdown('<span class="pill">📄 Document RAG Agent</span>', unsafe_allow_html=True)
    st.title("Ask your document anything")
    st.markdown(
        '<p class="muted">Upload a file, then ask questions — answers come only from the document.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Upload card ----
    with st.container(border=True):
        st.markdown("#### 📤 Upload a document")
        uploaded_file = st.file_uploader(
            "Drag and drop a file here, or click to browse",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "py", "ipynb", "pptx", "ppt"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_sig = (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)

            if st.session_state.get("file_sig") != file_sig:
                with st.spinner("Reading and indexing document..."):
                    st.session_state.vectorstore = build_vectorstore(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        chunk_size,
                        chunk_overlap,
                    )
                    st.session_state.file_sig = file_sig
                    st.session_state.messages = []
            st.success(f"✅ Indexed **{uploaded_file.name}** — ask away below.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Chat interface ----
    if "vectorstore" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question about the document...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            chain, retriever = get_chain(
                st.session_state.vectorstore, model_name, top_k
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chain.invoke(question)
                    st.markdown(answer)

                with st.expander("🔍 Sources (retrieved chunks)"):
                    for doc in retriever.invoke(question):
                        page = doc.metadata.get("page", "?")
                        page_label = page + 1 if isinstance(page, int) else page
                        st.markdown(f"**Page {page_label}**")
                        st.text(doc.page_content[:500])
                        st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("👆 Upload a file above to get started.")


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if st.session_state.stage == "login":
    render_login()
else:
    render_app()
