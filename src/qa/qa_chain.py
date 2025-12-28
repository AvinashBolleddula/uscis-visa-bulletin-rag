# src/qa/qa_chain.py
"""
USCIS / Visa Bulletin QA Chain (Retrieval + Grounded Answering)

- Loads an existing persisted Chroma collection (V4 by default)
- Routes queries to the best chunk type (table_row / table_summary / text)
- Retrieves top-K chunks with optional metadata filters
- Generates a grounded answer with Gemini (Vertex AI) + citations

Usage:
    from src.qa.qa_chain import USCISQASystem

    qa = USCISQASystem(
        persist_dir="/content/drive/MyDrive/uscis-rag/chroma",
        collection_name="visa_bulletins_v4_hierarchical",
        embed_model_id="text-embedding-005",
        llm_model_id="gemini-2.0-flash",
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        region=os.getenv("GOOGLE_CLOUD_REGION"),
    )

    result = qa.ask(
        "F2A Final Action Date for Mexico November 2025",
        where={"source": "visabulletin_November2025.pdf"},
        k=8,
    )
    print(result["answer"])
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Types
# ----------------------------

class RetrievedDoc(TypedDict):
    citation: str
    metadata: Dict[str, Any]
    snippet: str


class QAResult(TypedDict):
    query: str
    intent: str
    used_filter: Optional[Dict[str, Any]]
    answer: str
    retrieved: List[RetrievedDoc]


# ----------------------------
# Helpers
# ----------------------------

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _is_table_query(q: str) -> bool:
    ql = q.lower()
    # category codes (common)
    has_category = bool(re.search(r"\b(f1|f2a|f2b|f3|f4|eb-?1|eb-?2|eb-?3|eb-?4|eb-?5)\b", ql))
    has_country = any(c in ql for c in ["china", "india", "mexico", "philippines", "all countries", "all chargeability"])
    has_date_words = any(w in ql for w in ["final action", "dates", "date", "current", "unavailable", "priority"])
    return has_category and (has_country or has_date_words)


def detect_query_intent(q: str) -> str:
    """
    Decide which V4 content_type to prioritize.
    Returns one of: "table_row" | "table_summary" | "text"
    """
    ql = q.lower()

    if any(w in ql for w in ["summarize table", "table summary", "what does this table show"]):
        return "table_summary"

    if _is_table_query(q):
        return "table_row"

    return "text"


def _and_filter(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two Chroma filters with $and (without nesting too deeply)."""
    if not a:
        return b
    if not b:
        return a

    # If either already has $and, merge nicely
    if "$and" in a:
        left = a["$and"]
        return {"$and": left + [b]}
    if "$and" in b:
        right = b["$and"]
        return {"$and": [a] + right}

    return {"$and": [a, b]}


def format_citation(doc: Document) -> str:
    """
    Human-readable provenance line.
    We rely on metadata you already store in V4:
      source, page, content_type, section_path, category, month, table_id, row_id
    """
    m = doc.metadata or {}
    src = m.get("source", "unknown_source")
    page = m.get("page", "unknown_page")
    ctype = m.get("content_type", "unknown_type")
    month = m.get("month", "")
    section = m.get("section_path", "")
    cat = m.get("category", "")
    table_id = m.get("table_id", "")
    row_id = m.get("row_id", "")

    parts = [f"{src}", f"p.{page}", f"type={ctype}"]
    if month:
        parts.append(f"month={month}")
    if cat:
        parts.append(f"cat={cat}")
    if section:
        parts.append(f"section={section}")
    if table_id:
        parts.append(f"table={table_id}")
    if row_id != "":
        parts.append(f"row={row_id}")
    return " | ".join(parts)


def build_context(docs: List[Document], max_chars: int = 12000) -> str:
    """
    Build a bounded context string with explicit [DOC #] tags for citations.
    """
    blocks: List[str] = []
    total = 0

    for i, doc in enumerate(docs, start=1):
        cite = format_citation(doc)
        text = (_safe_str(doc.page_content)).strip()
        block = f"[DOC {i}] {cite}\n{text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(blocks)


# ----------------------------
# Main QA System
# ----------------------------

@dataclass
class USCISQASystem:
    persist_dir: str
    collection_name: str
    embed_model_id: str = "text-embedding-005"
    llm_model_id: str = "gemini-2.0-flash"
    project_id: Optional[str] = None
    region: Optional[str] = None

    # internal
    _vectorstore: Optional[Chroma] = None
    _llm: Optional[GenerativeModel] = None

    def init(self) -> None:
        """
        Initialize Vertex and load the persisted vectorstore.
        Call once at startup.
        """
        project = self.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        region = self.region or os.getenv("GOOGLE_CLOUD_REGION")

        if not project or not region:
            raise ValueError(
                "Missing GCP config. Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_REGION "
                "or pass project_id/region to USCISQASystem."
            )

        vertexai.init(project=project, location=region)

        embeddings = VertexAIEmbeddings(model_name=self.embed_model_id)

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
        )

        # Lightweight sanity check
        try:
            _ = self._vectorstore._collection.count()
        except Exception as e:
            raise RuntimeError(f"Failed to load Chroma collection '{self.collection_name}' from {self.persist_dir}: {e}")

        self._llm = GenerativeModel(self.llm_model_id)

    @property
    def vectorstore(self) -> Chroma:
        if not self._vectorstore:
            raise RuntimeError("USCISQASystem not initialized. Call qa.init() first.")
        return self._vectorstore

    @property
    def llm(self) -> GenerativeModel:
        if not self._llm:
            raise RuntimeError("USCISQASystem not initialized. Call qa.init() first.")
        return self._llm

    def retrieve(
        self,
        query: str,
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
        prefer_intent: Optional[str] = None,
    ) -> tuple[str, Optional[Dict[str, Any]], List[Document]]:
        """
        Retrieve documents from V4 with intent routing + fallback broad search.
        Returns: (intent, used_filter, docs)
        """
        intent = prefer_intent or detect_query_intent(query)

        # Route by content_type first
        routed = {"content_type": intent}
        routed_filter = _and_filter(routed, where) if where else routed

        docs = self.vectorstore.similarity_search(query, k=k, filter=routed_filter)

        # Fallback: broaden if too few hits
        if len(docs) < max(2, k // 3):
            docs = self.vectorstore.similarity_search(query, k=k, filter=where)
            return intent, where, docs

        return intent, routed_filter, docs

    def answer(
        self,
        query: str,
        docs: List[Document],
        max_context_chars: int = 12000,
    ) -> str:
        """
        Generate a grounded answer (must cite [DOC #]).
        """
        context = build_context(docs, max_chars=max_context_chars)

        system_rules = """You are a USCIS/Visa Bulletin assistant.
RULES (must follow):
- Use ONLY the provided DOCUMENT EXCERPTS. Do not use outside knowledge.
- Do not invent dates, categories, or country values.
- If the answer is not in the excerpts, say "I don’t have enough information in the retrieved excerpts."
- When stating any specific fact, include citations like [DOC 2] or [DOC 1][DOC 3].
- If there are conflicting values, explicitly say so and cite both.
OUTPUT FORMAT:
1) Answer (2–6 sentences, concise)
2) Evidence (bullets with citations)
"""

        prompt = f"""{system_rules}

USER QUESTION:
{query}

DOCUMENT EXCERPTS:
{context}
"""

        resp = self.llm.generate_content(prompt)
        return _safe_str(getattr(resp, "text", "")).strip()

    def ask(
        self,
        query: str,
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
        max_context_chars: int = 12000,
    ) -> QAResult:
        """
        End-to-end: route -> retrieve -> answer -> return answer + retrieved snippets.
        """
        intent, used_filter, docs = self.retrieve(query=query, k=k, where=where)

        answer_text = self.answer(query=query, docs=docs, max_context_chars=max_context_chars)

        retrieved: List[RetrievedDoc] = []
        for d in docs:
            retrieved.append(
                {
                    "citation": format_citation(d),
                    "metadata": d.metadata or {},
                    "snippet": (_safe_str(d.page_content))[:600],
                }
            )

        return {
            "query": query,
            "intent": intent,
            "used_filter": used_filter,
            "answer": answer_text,
            "retrieved": retrieved,
        }


# ----------------------------
# Convenience constructor (optional)
# ----------------------------

def build_default_qa() -> USCISQASystem:
    """
    Convenience for your current setup.
    Adjust persist_dir if you run locally (recommended).
    """
    qa = USCISQASystem(
        persist_dir="/Users/avinashbolleddula/Library/CloudStorage/GoogleDrive-avinash.bolleddula@gmail.com/My Drive/uscis-rag/chroma",
        collection_name="visa_bulletins_v4_hierarchical",
        embed_model_id="text-embedding-005",
        llm_model_id=os.getenv("QA_LLM_MODEL", "gemini-2.0-flash"),
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        region=os.getenv("GOOGLE_CLOUD_REGION"),
    )
    qa.init()
    return qa


if __name__ == "__main__":
    # Simple manual test (run: python -m src.qa.qa_chain)
    qa = build_default_qa()
    out = qa.ask(
        "F2A Final Action Date for Mexico November 2025",
        where={"source": "visabulletin_November2025.pdf"},
        k=8,
    )
    print(out["answer"])
    print("\nTop retrieved:")
    for r in out["retrieved"][:3]:
        print("-", r["citation"])