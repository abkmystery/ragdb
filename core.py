from __future__ import annotations

"""
core.py

Core implementation of the RAGdb embedded document database.
Unified System for Automated Multimodal Ingestion and Portable Knowledge Storage.
"""

import hashlib
import json
import math
import os
import pickle
import re
import sqlite3
import warnings
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any

import numpy as np
import csv as _csv

# SOTA OCR Check
try:
    from rapidocr_onnxruntime import RapidOCR

    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class DocumentRecord:
    path: str
    media_type: str
    mime_type: str
    content: str
    metadata: Dict
    preview: str
    file_hash: str  # Added for incremental updates

@dataclass
class SearchResult:
    path: str
    score: float
    media_type: str
    preview: str

class RAGdb:
    """
    Embedded document store with incremental ingestion and TF–IDF search.
    Acts as a portable single-file knowledge container.
    """

    def __init__(self, db_path: str = "RAGdb.ragdb") -> None:
        self.db_path = str(db_path)
        self._vectorizer_path = self.db_path + ".vectorizer.pkl"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Faster concurrent access
        self._create_tables()

        # Initialize OCR once if available
        self._ocr_engine = RapidOCR() if _HAS_OCR else None

    # ---------------- Schema ----------------
    def _create_tables(self) -> None:
        with closing(self.conn.cursor()) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    path       TEXT UNIQUE NOT NULL,
                    file_hash  TEXT,
                    media_type TEXT NOT NULL,
                    mime_type  TEXT,
                    content    TEXT NOT NULL,
                    vector     BLOB,
                    metadata   TEXT,
                    preview    TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
        self.conn.commit()

    # ---------------- Public API ----------------
    def ingest_file(self, file_path: str) -> None:
        self.ingest(file_path)

    def ingest_folder(self, folder_path: str) -> None:
        self.ingest(folder_path)

    def delete_file(self, file_path: str) -> None:
        norm = str(Path(file_path).resolve())
        with closing(self.conn.cursor()) as cur:
            cur.execute("DELETE FROM documents WHERE path = ?", (norm,))
            deleted = cur.rowcount
        self.conn.commit()
        print(f"[RAGdb] {'Deleted' if deleted else 'Not found'}: {norm}")
        if deleted:
            # Re-optimize vectors after deletion
            self._rebuild_vectors()

    def list_documents(self, limit: int = 50):
        with closing(self.conn.cursor()) as cur:
            cur.execute(
                "SELECT path, media_type, updated_at FROM documents ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    # ---------------- Ingestion (SOTA Incremental) ----------------
    def ingest(self, input_path: str) -> None:
        """
        Ingests files. Checks hashes to skip unchanged files (Incremental).
        """
        files = self._collect_files(input_path)
        if not files:
            print(f"[RAGdb] No supported files found at: {input_path}")
            return

        # Load existing hashes to avoid re-processing
        existing_hashes = {}
        with closing(self.conn.cursor()) as cur:
            cur.execute("SELECT path, file_hash FROM documents")
            for p, h in cur.fetchall():
                existing_hashes[p] = h

        updates_made = False
        processed_records = []

        for f in files:
            p = Path(f)
            current_hash = self._compute_hash(p)
            path_str = str(p.resolve())

            # Skip if unchanged
            if path_str in existing_hashes and existing_hashes[path_str] == current_hash:
                continue

            try:
                rec = self._extract(p)
                rec.file_hash = current_hash
                processed_records.append(rec)
                updates_made = True
                print(f"[RAGdb] Processed: {p.name}")
            except Exception as exc:
                print(f"[RAGdb] Error processing {f}: {exc}")

        if not updates_made and not existing_hashes:
            print("[RAGdb] Nothing to index.")
            return

        if not updates_made:
            print("[RAGdb] All files up to date.")
            return

        # Insert new/updated records into DB (temporarily without vectors)
        with closing(self.conn.cursor()) as cur:
            for rec in processed_records:
                # Upsert logic
                cur.execute(
                    """
                    INSERT INTO documents (path, file_hash, media_type, mime_type, content, vector, metadata, preview, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        file_hash=excluded.file_hash,
                        content=excluded.content,
                        vector=excluded.vector,
                        metadata=excluded.metadata,
                        updated_at=excluded.updated_at
                    """,
                    (
                        rec.path, rec.file_hash, rec.media_type, rec.mime_type,
                        rec.content, None,  # Vector computed next
                        json.dumps(rec.metadata, ensure_ascii=False),
                        rec.preview, _now_iso()
                    ),
                )
        self.conn.commit()

        # Rebuild vectors for EVERYONE to keep TF-IDF space consistent
        self._rebuild_vectors()

    def _rebuild_vectors(self):
        """Reads all content, rebuilds vocab, updates vectors."""
        docs = {}
        with closing(self.conn.cursor()) as cur:
            cur.execute("SELECT id, content FROM documents")
            for row in cur.fetchall():
                docs[row[0]] = row[1]

        if not docs: return

        print(f"[RAGdb] Updating vector space for {len(docs)} documents...")
        vocab, idf = self._build_vectorizer(list(docs.values()))

        # Save vectorizer model
        with open(self._vectorizer_path, "wb") as f:
            pickle.dump({"vocab": vocab, "idf": idf}, f)

        # Update blobs
        with closing(self.conn.cursor()) as cur:
            for doc_id, text in docs.items():
                vec = self._compute_vector(text, vocab, idf)
                cur.execute("UPDATE documents SET vector = ? WHERE id = ?",
                            (vec.astype("float32").tobytes(), doc_id))
        self.conn.commit()

    # ---------------- Search ----------------
    def search(self, query: str, top_k: int = 5):
        """
        SOTA Hybrid Search: Combines TF-IDF Vector similarity with
        exact substring matching. Finds partial words and exact phrases.
        """
        if not os.path.exists(self._vectorizer_path):
            raise RuntimeError("Index not found. Run ingest() first.")

        with open(self._vectorizer_path, "rb") as f:
            data = pickle.load(f)

        vocab, idf = data["vocab"], data["idf"]
        q_vec = self._compute_vector(query, vocab, idf)

        # Pre-compute query for substring check
        q_clean = query.lower().strip()

        results = []
        with closing(self.conn.cursor()) as cur:
            # Fetch 'content' as well to perform the substring check
            cur.execute("SELECT path, media_type, preview, vector, content FROM documents WHERE vector IS NOT NULL")

            for path, media_type, preview, blob, content in cur.fetchall():
                score = 0.0

                # 1. Vector Score (Semantic Match)
                d_vec = np.frombuffer(blob, dtype="float32")
                if d_vec.size == q_vec.size:
                    dot = float(np.dot(q_vec, d_vec))
                    score += dot

                # 2. Substring Boost (Keyword Match)
                # If the query is actually inside the text, give it a massive boost.
                # This fixes the "kan" vs "kaneez" issue.
                if q_clean and q_clean in content.lower():
                    score += 1.0

                if score > 0:
                    results.append(SearchResult(path, score, media_type, preview or ""))

        # Sort by highest score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ---------------- Helpers ----------------
    def _compute_hash(self, path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    def _collect_files(self, path: str):
        p = Path(path)
        if p.is_file():
            return [str(p.resolve())] if self._is_supported(p.suffix.lower()) else []
        out = []
        if p.is_dir():
            for fp in p.rglob("*"):
                if fp.is_file() and self._is_supported(fp.suffix.lower()):
                    out.append(str(fp.resolve()))
        return out

    @staticmethod
    def _is_supported(suffix: str) -> bool:
        return suffix in {
            ".txt", ".pdf", ".docx", ".json", ".csv", ".xls", ".xlsx",
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
            ".wav", ".mp3", ".ogg", ".flac", ".m4a",
            ".mp4", ".mov", ".mkv", ".avi", ".webm"
        }

    # ----- Extraction logic (Modified for RapidOCR) -----
    def _extract(self, path: Path) -> DocumentRecord:
        sfx = path.suffix.lower()
        if sfx == ".txt": return self._extract_text(path)
        if sfx == ".pdf": return self._extract_pdf(path)
        if sfx == ".docx": return self._extract_docx(path)
        if sfx == ".json": return self._extract_json(path)
        if sfx == ".csv": return self._extract_csv(path)
        if sfx in {".xls", ".xlsx"}: return self._extract_excel(path)
        if sfx in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}: return self._extract_image(path)
        if sfx in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}: return self._extract_audio(path)
        if sfx in {".mp4", ".mov", ".mkv", ".avi", ".webm"}: return self._extract_video(path)
        raise ValueError(f"Unsupported: {sfx}")

    def _extract_text(self, path: Path) -> DocumentRecord:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        return self._make_rec(path, "text", "text/plain", text, {})

    def _extract_pdf(self, path: Path) -> DocumentRecord:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return self._make_rec(path, "pdf", "application/pdf", text, {"pages": len(reader.pages)})

    def _extract_docx(self, path: Path) -> DocumentRecord:
        import docx
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return self._make_rec(path, "docx", "application/vnd.openxmlformats", text, {})

    def _extract_json(self, path: Path) -> DocumentRecord:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        text = json.dumps(data, indent=2, ensure_ascii=False)
        return self._make_rec(path, "json", "application/json", text, {})

    def _extract_csv(self, path: Path) -> DocumentRecord:
        rows = []
        with path.open(encoding="utf-8-sig", errors="ignore") as f:
            reader = _csv.reader(f)
            for i, r in enumerate(reader):
                rows.append(" ".join(r))
                if i > 5000: break  # Limit large CSVs
        return self._make_rec(path, "csv", "text/csv", "\n".join(rows), {"rows": len(rows)})

    def _extract_excel(self, path: Path) -> DocumentRecord:
        import pandas as pd
        df = pd.read_excel(path)
        text = df.to_csv(index=False)
        return self._make_rec(path, "excel", "application/vnd.ms-excel", text, {"rows": len(df)})

    def _extract_image(self, path: Path) -> DocumentRecord:
        # SOTA: RapidOCR implementation
        text = ""
        meta = {"size_bytes": path.stat().st_size}

        if self._ocr_engine:
            try:
                # result is list of [box, text, score]
                result, _ = self._ocr_engine(str(path))
                if result:
                    lines = [line[1] for line in result]
                    text = "\n".join(lines)
                    meta["ocr_engine"] = "RapidOCR"
            except Exception as e:
                print(f"[RAGdb] OCR Error on {path.name}: {e}")

        # Fallback description
        if not text:
            text = f"Image file: {path.name}"

        return self._make_rec(path, "image", "image/*", text, meta)

    def _extract_audio(self, path: Path) -> DocumentRecord:
        # Metadata only for now
        return self._make_rec(path, "audio", "audio/*", f"Audio file: {path.name}", {})

    def _extract_video(self, path: Path) -> DocumentRecord:
        return self._make_rec(path, "video", "video/*", f"Video file: {path.name}", {})

    def _make_rec(self, path: Path, mtype: str, mime: str, content: str, meta: Dict) -> DocumentRecord:
        preview = content.strip().replace("\n", " ")[:500]
        return DocumentRecord(
            path=str(path.resolve()),
            media_type=mtype,
            mime_type=mime,
            content=content,
            metadata=meta,
            preview=preview,
            file_hash=""
        )

    # ----- Vectorizer (Improved TF-IDF) -----
    def _tokenize(self, text: str):
        return [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_]+\b", text) if len(t) > 1]

    def _build_vectorizer(self, texts: Sequence[str]):
        # Build vocabulary
        doc_count = len(texts)
        df_counts: Dict[str, int] = {}

        for txt in texts:
            seen = set(self._tokenize(txt))
            for tok in seen:
                df_counts[tok] = df_counts.get(tok, 0) + 1

        # Prune rare words (noise reduction)
        sorted_vocab = sorted([k for k, v in df_counts.items() if v > 0])
        vocab = {tok: i for i, tok in enumerate(sorted_vocab)}

        # Compute IDF
        idf = np.zeros(len(vocab), dtype="float32")
        for tok, idx in vocab.items():
            # Smoothed IDF
            idf[idx] = math.log((doc_count + 1) / (df_counts[tok] + 1)) + 1.0

        return vocab, idf

    def _compute_vector(self, text: str, vocab, idf):
        tokens = self._tokenize(text)
        vec = np.zeros(len(vocab), dtype="float32")

        # Term Frequency
        tf: Dict[str, int] = {}
        for t in tokens:
            if t in vocab:
                tf[t] = tf.get(t, 0) + 1

        for tok, count in tf.items():
            idx = vocab[tok]
            # Log-normalization for TF (sublinear scaling) - SOTA practice for TF-IDF
            tf_val = 1.0 + math.log(count)
            vec[idx] = tf_val * idf[idx]

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec