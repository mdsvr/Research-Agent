import os
import glob
import json
import re
from typing import List, Tuple, Dict, Any
from veritas.config import Config
from veritas.schemas import Chunk
from veritas.chunking import chunk_document

def load_file_content(file_path: str) -> str:
    """Read .txt, .md, or .pdf files and normalize text."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext in [".txt", ".md"]:
        raw = open(file_path, "rb").read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Never decode with errors="ignore": dropped bytes silently corrupt the
            # citation spans that the whole system is built on.
            print(f"Warning: {file_path} is not valid UTF-8; decoding as cp1252.")
            text = raw.decode("cp1252", errors="replace")
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text() or ""
                pages_text.append(t)
            text = "\n".join(pages_text)
        except Exception as e:
            print(f"Warning: Failed to parse PDF {file_path}: {e}")
            return ""
    else:
        return ""

    # Normalize whitespace and strip repeating artifacts
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_title(content: str, filename: str) -> str:
    """The document's own title — its first markdown H1 — falling back to the filename.

    This matters beyond cosmetics: the real titles carry the identifiers the body text
    refers to only implicitly ("Technical Analysis of CVE-2024-3094: ..." vs a chunk that
    just says "The vulnerability ..."). The verifier prepends this title to the premise,
    which is what lets a claim binding an identifier to a value be entailed at all.
    """
    for line in content.splitlines():
        if line.startswith("# "):
            heading = line[2:].strip()
            if heading:
                return heading
    return os.path.splitext(filename)[0].replace("_", " ").title()


def run_ingest(cfg: Config) -> Tuple[int, int]:
    """
    Scans corpus_dir, loads documents, generates chunks, writes manifest.json and chunk cache.
    Returns (total_chunks, total_docs).
    """
    os.makedirs(cfg.index_dir, exist_ok=True)
    corpus_files = sorted(glob.glob(os.path.join(cfg.corpus_dir, "*.*")))
    
    manifest: List[Dict[str, Any]] = []
    all_chunks: List[Chunk] = []

    doc_counter = 1
    for fpath in corpus_files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in [".txt", ".md", ".pdf"]:
            continue

        filename = os.path.basename(fpath)
        doc_id = f"S{doc_counter}"

        content = load_file_content(fpath)
        if not content:
            continue

        title = extract_title(content, filename)

        # Generate sentence-window chunks
        chunks = chunk_document(
            doc_id=doc_id,
            doc_title=title,
            text=content,
            target_tokens=cfg.chunk.target_tokens,
            overlap_sentences=cfg.chunk.overlap_sentences
        )

        all_chunks.extend(chunks)
        manifest.append({
            "doc_id": doc_id,
            "filename": filename,
            "title": title,
            "char_count": len(content),
            "chunk_count": len(chunks)
        })

        doc_counter += 1

    # Save manifest
    with open(cfg.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save chunks as JSON cache
    chunks_json_path = os.path.join(cfg.index_dir, "chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, indent=2)

    # Build retrieval indices (dense + sparse)
    from veritas.index import build_indices
    build_indices(all_chunks, cfg)

    return len(all_chunks), len(manifest)
