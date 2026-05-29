"""
In-memory face embedding cache for fast concurrent /match vector search.

Loaded at startup; updated incrementally on register. Avoids SELECT * FROM faces
on every match request.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from database import list_faces_for_matching, list_faces_for_matching_by_label


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


EMBEDDING_CACHE_ENABLED = _env_bool("EMBEDDING_CACHE_ENABLED", True)


@dataclass(frozen=True)
class CachedFaceRow:
    label: str
    source: str
    image_url: str
    doc_type: str
    embedding: np.ndarray  # L2-normalized float32 (512,)


class EmbeddingCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rows: List[CachedFaceRow] = []
        self._matrix: Optional[np.ndarray] = None
        self._label_index: Dict[str, List[int]] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded

    @property
    def face_count(self) -> int:
        with self._lock:
            return len(self._rows)

    def _rebuild_index(self, rows: List[CachedFaceRow]) -> None:
        self._rows = rows
        label_index: Dict[str, List[int]] = {}
        valid_matrix_rows: List[np.ndarray] = []
        for i, row in enumerate(rows):
            key = " ".join(str(row.label).split()).lower()
            label_index.setdefault(key, []).append(i)
            valid_matrix_rows.append(row.embedding)
        self._label_index = label_index
        self._matrix = (
            np.stack(valid_matrix_rows, axis=0).astype(np.float32, copy=False)
            if valid_matrix_rows
            else None
        )
        self._loaded = True

    @staticmethod
    def _row_from_doc(doc: Dict[str, Any]) -> Optional[CachedFaceRow]:
        emb = doc.get("embedding")
        if not emb or len(emb) != 512:
            return None
        vec = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return None
        return CachedFaceRow(
            label=str(doc.get("label") or ""),
            source=str(doc.get("source") or ""),
            image_url=str(doc.get("image_url") or ""),
            doc_type=str(doc.get("doc_type") or "Selfie"),
            embedding=(vec / norm).astype(np.float32, copy=False),
        )

    async def reload_from_db(self) -> int:
        docs = await list_faces_for_matching()
        rows: List[CachedFaceRow] = []
        for doc in docs:
            row = self._row_from_doc(doc)
            if row is not None:
                rows.append(row)
        with self._lock:
            self._rebuild_index(rows)
        return len(rows)

    def add_face(
        self,
        *,
        label: str,
        source: str,
        image_url: str,
        embedding: List[float],
        doc_type: str = "Selfie",
    ) -> None:
        row = self._row_from_doc(
            {
                "label": label,
                "source": source,
                "image_url": image_url,
                "doc_type": doc_type,
                "embedding": embedding,
            }
        )
        if row is None:
            return
        with self._lock:
            rows = list(self._rows)
            rows.append(row)
            self._rebuild_index(rows)

    def search_scores(
        self,
        emb: np.ndarray,
        *,
        scoped_label: Optional[str] = None,
        top_k: int = 50,
    ) -> Tuple[List[Dict[str, Any]], int, str]:
        """
        Returns (raw_result dicts sorted by score desc, candidate_count, source_note).
        """
        if not self._loaded:
            return [], 0, "cache_not_loaded"

        query = np.asarray(emb, dtype=np.float32).reshape(-1)
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0.0:
            return [], 0, "zero_query_embedding"
        query = query / qnorm

        with self._lock:
            if self._matrix is None or not self._rows:
                return [], 0, "cache_empty"

            if scoped_label:
                key = " ".join(str(scoped_label).split()).lower()
                indices = self._label_index.get(key, [])
                if not indices:
                    return [], 0, "scoped_label_miss"
                sub_matrix = self._matrix[indices]
                scores = sub_matrix @ query
                pairs = [(float(scores[i]), indices[i]) for i in range(len(indices))]
                pairs.sort(key=lambda x: x[0], reverse=True)
                pairs = pairs[:top_k]
                source_note = f"cache_scoped:{len(indices)}"
            else:
                scores = self._matrix @ query
                order = np.argsort(-scores)[:top_k]
                pairs = [(float(scores[i]), int(i)) for i in order]
                source_note = f"cache_open:{len(self._rows)}"

            out: List[Dict[str, Any]] = []
            for score, idx in pairs:
                row = self._rows[idx]
                out.append(
                    {
                        "label": row.label,
                        "source": row.source,
                        "image_url": row.image_url,
                        "doc_type": row.doc_type,
                        "score": round(score, 3),
                    }
                )
            return out, len(pairs), source_note

    async def fallback_search(
        self,
        emb: np.ndarray,
        *,
        scoped_label: Optional[str] = None,
        top_k: int = 50,
    ) -> Tuple[List[Dict[str, Any]], int, str]:
        """DB fallback when cache is disabled or empty."""
        if scoped_label:
            docs = await list_faces_for_matching_by_label(scoped_label)
            note = f"db_scoped:{len(docs)}"
        else:
            docs = await list_faces_for_matching()
            note = f"db_open:{len(docs)}"

        query = np.asarray(emb, dtype=np.float32).reshape(-1)
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0.0:
            return [], 0, "zero_query_embedding"
        query = query / qnorm

        raw: List[Dict[str, Any]] = []
        for doc in docs:
            row = self._row_from_doc(doc)
            if row is None:
                continue
            score = float(np.dot(row.embedding, query))
            raw.append(
                {
                    "label": row.label,
                    "source": row.source,
                    "image_url": row.image_url,
                    "doc_type": row.doc_type,
                    "score": round(score, 3),
                }
            )
        raw.sort(key=lambda x: x["score"], reverse=True)
        return raw[:top_k], len(raw), note


embedding_cache = EmbeddingCache()
