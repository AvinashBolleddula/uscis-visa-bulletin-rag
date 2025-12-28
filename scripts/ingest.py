#!/usr/bin/env python3
"""
Ingest Visa Bulletin PDFs into Chroma using V4 hierarchical chunking.

Usage:
  uv run python scripts/ingest.py \
    --data-dir data \
    --persist-dir ./chroma \
    --collection visa_bulletins_v4_hierarchical \
    --embed-model text-embedding-005 \
    --wipe

Env:
  GOOGLE_CLOUD_PROJECT
  GOOGLE_CLOUD_REGION
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import vertexai
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------------------
# Helpers: month extraction
# ---------------------------

def extract_month_from_filename(filename: str) -> str:
    month_map = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12",
    }
    for month_name, month_num in month_map.items():
        if month_name in filename:
            year = re.search(r"(\d{4})", filename)
            if year:
                return f"{year.group(1)}-{month_num}"
    return "unknown"


# ---------------------------
# V4: section hierarchy
# ---------------------------

@dataclass
class SectionNode:
    title: str
    level: int
    page_num: int
    parent: Optional["SectionNode"] = None
    children: List["SectionNode"] = None

    def __post_init__(self):
        self.title = self.title.strip()
        self.children = self.children or []

    def get_hierarchy_path(self) -> List[str]:
        path: List[str] = []
        node: Optional["SectionNode"] = self
        while node:
            path.insert(0, node.title)
            node = node.parent
        return path

    def get_section_path(self) -> str:
        return " > ".join(self.get_hierarchy_path())


def detect_section_headers(pdf_path: str) -> List[SectionNode]:
    sections: List[SectionNode] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue

            for line in text.split("\n"):
                s = line.strip()
                if not s:
                    continue

                # Level 1: "A. SOME TITLE"
                if re.match(r"^[A-Z]\.\s+[A-Z][A-Z\s\-\(\),]+$", s, re.IGNORECASE):
                    sections.append(SectionNode(s, level=1, page_num=page_num))
                # Level 2: short lines containing PREFERENCES
                elif "PREFERENCES" in s.upper() and len(s.split()) <= 6:
                    sections.append(SectionNode(s, level=2, page_num=page_num))
                # Level 3: lines containing DATES / ACTION
                elif re.match(r"^[A-Z]\.\s+.*(DATES|ACTION)", s, re.IGNORECASE):
                    sections.append(SectionNode(s, level=3, page_num=page_num))
                # Other main sections
                elif "DIVERSITY" in s.upper() and "IMMIGRANT" in s.upper():
                    sections.append(SectionNode(s, level=1, page_num=page_num))

    return sections


def build_section_tree(sections: List[SectionNode]) -> List[SectionNode]:
    if not sections:
        return []

    roots: List[SectionNode] = []
    stack: List[SectionNode] = []

    for sec in sections:
        while stack and stack[-1].level >= sec.level:
            stack.pop()

        if stack:
            sec.parent = stack[-1]
            stack[-1].children.append(sec)
        else:
            roots.append(sec)

        stack.append(sec)

    return roots


def flatten_tree(nodes: List[SectionNode]) -> List[SectionNode]:
    flat: List[SectionNode] = []
    for n in nodes:
        flat.append(n)
        flat.extend(flatten_tree(n.children))
    return flat


# ---------------------------
# V4: content extraction
# ---------------------------

def extract_text_for_section(pdf_path: str, section: SectionNode, next_section: Optional[SectionNode]) -> str:
    # NOTE: simple heuristic: extract on the section's page only
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[section.page_num - 1]
        text = page.extract_text() or ""

    lines = text.split("\n")
    collecting = False
    out: List[str] = []

    for line in lines:
        if section.title in line:
            collecting = True
            continue
        if collecting:
            if next_section and next_section.title in line:
                break
            if re.match(r"^[A-Z]\.\s+[A-Z\s]+", line.strip()):
                break
            out.append(line)

    return "\n".join(out).strip()


def extract_tables_for_section(pdf_path: str, section: SectionNode) -> List[Dict[str, Any]]:
    # NOTE: current notebook logic: tables on this section's page only
    tables: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[section.page_num - 1]
        raw_tables = page.extract_tables()

    for idx, t in enumerate(raw_tables or []):
        if t and len(t) > 1:
            tables.append({"table_idx": idx, "table_data": t, "page_num": section.page_num})

    return tables


# ---------------------------
# V4: row chunks
# ---------------------------

def identify_category(row: List[str]) -> Optional[str]:
    if not row:
        return None
    first = (str(row[0]).strip() if row[0] else "")
    if first in ["F1", "F2A", "F2B", "F3", "F4"]:
        return first
    if "EB-1" in first or first == "1st":
        return "EB-1"
    if "EB-2" in first or first == "2nd":
        return "EB-2"
    if "EB-3" in first or first == "3rd":
        return "EB-3"
    if "EB-4" in first or first == "4th" or "Certain Religious Workers" in first:
        return "EB-4"
    if "EB-5" in first or first == "5th":
        return "EB-5"
    return None


def normalize_date_or_status(value: str) -> str:
    if not value:
        return "unknown"
    v = str(value).strip().upper()
    if v in ["C", "CURRENT"]:
        return "Current"
    if v in ["U", "UNAVAILABLE"]:
        return "Unavailable"
    return v


def generate_templated_summary(metadata: Dict[str, Any]) -> str:
    month = metadata.get("month", "unknown")
    section = str(metadata.get("section", "unknown"))
    category = metadata.get("category", "unknown")

    countries_map = {
        "All Countries": "all_countries",
        "China": "china",
        "India": "india",
        "Mexico": "mexico",
        "Philippines": "philippines",
    }

    parts = []
    for cname, key in countries_map.items():
        val = metadata.get(key, "unknown")
        if val != "unknown":
            parts.append(
                f"Visa Bulletin {month} — {section} {category} — {cname} — Final Action Date: {val}"
            )
    return " | ".join(parts)


def build_row_chunk(
    row: List[str],
    header_row: List[str],
    month: str,
    section_path: str,
    page_num: int,
    table_id: str,
    row_id: int,
    source_file: str,
) -> Optional[Document]:
    if not row or all(not c for c in row):
        return None

    category = identify_category(row)
    if not category:
        return None

    row_text = " | ".join([str(c).strip() if c else "" for c in row])
    header_text = " | ".join([str(c).strip() if c else "" for c in header_row])

    countries = ["All", "China", "India", "Mexico", "Philippines"]
    country_dates: Dict[str, str] = {}
    for i, c in enumerate(countries):
        idx = i + 1
        if idx < len(row):
            country_dates[c] = normalize_date_or_status(row[idx])

    metadata = {
        "source": source_file,
        "month": month,
        "section_path": section_path,
        "category": category,
        "page": page_num,
        "table_id": table_id,
        "row_id": row_id,
        "content_type": "table_row",
        "all_countries": country_dates.get("All", "unknown"),
        "china": country_dates.get("China", "unknown"),
        "india": country_dates.get("India", "unknown"),
        "mexico": country_dates.get("Mexico", "unknown"),
        "philippines": country_dates.get("Philippines", "unknown"),
    }

    content = f"{generate_templated_summary(metadata)}\n\nHEADER: {header_text}\nDATA: {row_text}"
    return Document(page_content=content, metadata=metadata)


# ---------------------------
# Main: build V4 chunks
# ---------------------------

def build_v4_chunks(pdf_path: Path) -> List[Document]:
    source_file = pdf_path.name
    month = extract_month_from_filename(source_file)

    sections = detect_section_headers(str(pdf_path))
    tree = build_section_tree(sections)
    all_sections = flatten_tree(tree)

    chunks: List[Document] = []

    # text splitter for narrative
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for i, sec in enumerate(all_sections):
        section_path = sec.get_section_path()
        next_sec = all_sections[i + 1] if i + 1 < len(all_sections) else None

        # TEXT chunks
        txt = extract_text_for_section(str(pdf_path), sec, next_sec)
        if txt and len(txt) > 50:
            docs = splitter.create_documents([txt])
            for d in docs:
                d.metadata.update(
                    {
                        "source": source_file,
                        "month": month,
                        "content_type": "text",
                        "section_path": section_path,
                        "page": sec.page_num,
                    }
                )
                chunks.append(d)

        # TABLE chunks (row-level)
        tables = extract_tables_for_section(str(pdf_path), sec)
        for t in tables:
            table_data = t["table_data"]
            if len(table_data) < 2:
                continue

            header = table_data[0]
            table_id = f"table_{sec.page_num}_{t['table_idx']}"

            for row_idx, row in enumerate(table_data[1:], start=1):
                doc = build_row_chunk(
                    row=row,
                    header_row=header,
                    month=month,
                    section_path=section_path,
                    page_num=sec.page_num,
                    table_id=table_id,
                    row_id=row_idx,
                    source_file=source_file,
                )
                if doc:
                    chunks.append(doc)

    return chunks


# ---------------------------
# Ingest
# ---------------------------

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--persist-dir", type=str, default="./chroma")
    parser.add_argument("--collection", type=str, default="visa_bulletins_v4_hierarchical")
    parser.add_argument("--embed-model", type=str, default="text-embedding-005")
    parser.add_argument("--wipe", action="store_true", help="Delete existing collection first")
    parser.add_argument("--glob", type=str, default="visabulletin*.pdf")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    region = os.getenv("GOOGLE_CLOUD_REGION")
    if not project or not region:
        raise ValueError("Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_REGION in .env")

    vertexai.init(project=project, location=region)

    data_dir = Path(args.data_dir)
    pdf_paths = sorted(data_dir.glob(args.glob))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir} matching {args.glob}")

    print(f"Found {len(pdf_paths)} PDFs")

    # Build chunks
    all_docs: List[Document] = []
    for p in pdf_paths:
        print(f"📄 Chunking: {p.name}")
        docs = build_v4_chunks(p)
        print(f"   -> {len(docs)} chunks")
        all_docs.extend(docs)

    print(f"Total chunks: {len(all_docs)}")

    embeddings = VertexAIEmbeddings(model_name=args.embed_model)

    # Init vectorstore
    vs = Chroma(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
    )

    if args.wipe:
        try:
            vs.delete_collection()
            print(f"✅ Deleted existing collection: {args.collection}")
        except Exception:
            print("ℹ️ No collection to delete (or delete failed). Continuing...")

        vs = Chroma(
            collection_name=args.collection,
            persist_directory=args.persist_dir,
            embedding_function=embeddings,
        )

    # Ingest in batches
    bs = args.batch_size
    for i in range(0, len(all_docs), bs):
        batch = all_docs[i : i + bs]
        vs.add_documents(batch)
        print(f"✅ Ingested batch {i//bs + 1} ({len(batch)} docs)")

    # Persist to disk
    vs.persist()
    count = vs._collection.count()
    print(f"\n✅ Done. Persisted count: {count}")
    print(f"Chroma dir: {args.persist_dir}")
    print(f"Collection: {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())