# ui/app.py
from __future__ import annotations

# read env vars + default paths
import os
# pretty print metadata in the UI
import json
# Streamlit UI, build the web ui
import streamlit as st
# load .env into environment variables
from dotenv import load_dotenv

# add repo root to python path so from src... works
# when running streamlit run ui/app.py
import sys
from pathlib import Path

# for running subprocesses, here used to sync GCS
import subprocess

import time
# add repo root to python path so `import src...` works
# Streamlit runs ui/app.py as the entrypoint, 
# so Python may not know where src/ is. This makes import src.qa... work.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import our QA system, brings in your retrial + answering logic
from src.qa.qa_chain import USCISQASystem

# load env vars from .env
load_dotenv()

# sets the Streamlit page title + layout
st.set_page_config(page_title="USCIS Visa Bulletin RAG", layout="wide")

# shows title + description
st.title("USCIS Visa Bulletin RAG (V4)")
st.caption("Ask questions over Visa Bulletin PDFs using Chroma + Vertex AI (Gemini).")

# -------------------------
# Sidebar: Config
# -------------------------

# Everything inside with st.sidebar: is just inputs for runtime config:
with st.sidebar:
    st.subheader("Sync Chroma from GCS")

    gcs_chroma_path = st.text_input(
        "GCS_CHROMA_PATH",
        value=os.getenv("GCS_CHROMA_PATH", ""),
        placeholder="gs://bucket/prefix/chroma",
        help="GCS folder that contains the Chroma persisted directory.",
    )

    if st.button("⬇️ Update from GCS", use_container_width=True):
        if not gcs_chroma_path.strip():
            st.error("Set GCS_CHROMA_PATH (e.g., gs://bucket/prefix/chroma).")
        else:
            with st.spinner("Pulling Chroma directory from GCS..."):
                try:
                    cmd = [
                        sys.executable,  # uses same venv python under uv
                        "scripts/gcs_sync.py",
                        "--mode",
                        "pull",
                        "--local",
                        os.getenv("CHROMA_PERSIST_DIR", "./chroma"),
                        "--gcs",
                        gcs_chroma_path.strip(),
                    ]
                    subprocess.check_call(cmd)

                    # Clear cached QA object so it reloads from refreshed persist_dir
                    st.cache_resource.clear()

                    st.success("✅ Updated local Chroma from GCS. Reloading…")
                    st.rerun()
                except subprocess.CalledProcessError as e:
                    st.error("GCS sync failed.")
                    st.code(str(e))
    st.divider()
    st.header("Configuration")

    persist_dir = st.text_input(
        "Chroma persist dir",
        value=os.getenv("CHROMA_PERSIST_DIR", "./chroma"),
        help="Local path to your persisted Chroma directory (where collections live).",
    )

    collection_name = st.text_input(
        "Collection name",
        value=os.getenv("CHROMA_COLLECTION", "visa_bulletins_v4_hierarchical"),
    )

    embed_model = st.text_input(
        "Embedding model",
        value=os.getenv("EMBED_MODEL_ID", "text-embedding-005"),
    )

    llm_model = st.text_input(
        "LLM model",
        value=os.getenv("QA_LLM_MODEL", "gemini-2.0-flash"),
    )

    project_id = st.text_input(
        "GCP Project",
        value=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
    )

    region = st.text_input(
        "GCP Region",
        value=os.getenv("GOOGLE_CLOUD_REGION", ""),
    )

    st.divider()

    k = st.slider("Top K retrieved", min_value=3, max_value=15, value=5, step=1)

    st.subheader("Optional filters (metadata)")
    source = st.text_input("source (PDF filename)", value="", placeholder="visabulletin_November2025.pdf")
    month = st.text_input("month (YYYY-MM)", value="", placeholder="2025-11")
    category = st.text_input("category", value="", placeholder="F2A / EB-2 / ...")

    st.caption("Filters are applied as a Chroma `where` clause when provided.")
    


# Build where filter
# Creates the where dict you pass to qa.ask(...). If empty, you pass None.
where = {}
if source.strip():
    where["source"] = source.strip()
if month.strip():
    where["month"] = month.strip()
if category.strip():
    where["category"] = category.strip()

# helper function
# Check if CHROMA_PERSIST_DIR exists and is non-empty
# If not → run gcs_sync.py --mode pull
# Then continue with normal QA initialization
def ensure_chroma_available(persist_dir: str, gcs_path: str) -> None:
    """
    If persist_dir is missing or empty, pull Chroma from GCS.
    """
    p = Path(persist_dir)

    # If directory exists and has files, assume it's ready
    if p.exists() and any(p.iterdir()):
        return

    if not gcs_path:
        raise RuntimeError(
            f"Chroma directory '{persist_dir}' is empty and GCS_CHROMA_PATH is not set."
        )

    p.mkdir(parents=True, exist_ok=True)

    st.info("📥 First Streamlit Run, So Local Chroma not found. Synced from GCS...")
    subprocess.check_call(
        [
            "python",
            "scripts/gcs_sync.py",
            "--mode",
            "pull",
            "--local",
            persist_dir,
            "--gcs",
            gcs_path,
        ]
    )
    

# run to check + pull if needed
gcs_chroma_path = os.getenv("GCS_CHROMA_PATH", "")

try:
    ensure_chroma_available(persist_dir, gcs_chroma_path)
except Exception as e:
    st.error("Failed to prepare Chroma vector store.")
    st.code(str(e))
    st.stop()

# -------------------------
# Initialize QA
# -------------------------
# Caches the loaded QA system so it doesn't reload on every interaction
# Creating embeddings + loading Chroma + initializing Vertex should happen once, not on every button click or rerun
# st.cache_resource caches the initialized object so the app stays fast
@st.cache_resource(show_spinner=False)
# 
def load_qa(
    persist_dir: str,
    collection_name: str,
    embed_model: str,
    llm_model: str,
    project_id: str,
    region: str,
) -> USCISQASystem:
    qa = USCISQASystem(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embed_model_id=embed_model,
        llm_model_id=llm_model,
        project_id=project_id or None,
        region=region or None,
    )
    qa.init()
    return qa

qa = None
init_error = None
try:
    qa = load_qa(persist_dir, collection_name, embed_model, llm_model, project_id, region)
except Exception as e:
    init_error = str(e)

if init_error:
    st.error("QA system failed to initialize.")
    st.code(init_error)
    st.stop()

# -------------------------
# Main: Ask
# -------------------------
query = st.text_area(
    "Your question",
    value="F2A Final Action Date for Mexico November 2025",
    height=90,
)

colA, colB = st.columns([1, 3])
with colA:
    run = st.button("Ask", type="primary", use_container_width=True)
with colB:
    st.write("")

if run:
    with st.spinner("Retrieving + answering..."):
        
        out = qa.ask(
            query=query.strip(),
            k=k,
            where=where if where else None,
        )

    st.subheader("Answer")
    st.write(out["answer"] or "No answer returned.")

    st.subheader("Retrieval details")
    st.write(
        {
            "intent": out.get("intent"),
            "used_filter": out.get("used_filter"),
            "k": k,
        }
    )

    retrieved = out.get("retrieved", [])

    st.subheader(f"Top retrieved chunks ({len(retrieved)})")
    for i, r in enumerate(retrieved, start=1):
        with st.expander(f"[{i}] {r.get('citation','(no citation)')}"):
            st.markdown("**Snippet**")
            st.code(r.get("snippet", ""), language="text")

            st.markdown("**Metadata**")
            st.code(json.dumps(r.get("metadata", {}), indent=2), language="json")