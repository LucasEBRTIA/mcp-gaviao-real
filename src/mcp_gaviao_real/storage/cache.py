"""Page-granular SQLite + FTS5 cache and WebP image blob store."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from mcp_gaviao_real.config import GaviaoSettings, validate_file_path
from mcp_gaviao_real.models import DocumentInspection, ImageRef, SearchHit


class DocumentCache:
    """SQLite (WAL + FTS5) cache indexed by document fingerprint, page, and engine."""

    def __init__(self, cache_dir: Path, max_cache_bytes: int = 2 * 1024**3) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_cache_bytes = max_cache_bytes
        self.blobs_dir = self.cache_dir / "blobs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / "index.db"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                last_path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                format TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                toc_json TEXT NOT NULL,
                inspection_json TEXT,
                last_accessed REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_fast_fp
                ON documents(last_path, mtime_ns, file_size);

            CREATE TABLE IF NOT EXISTS pages (
                doc_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                engine TEXT NOT NULL,
                markdown TEXT NOT NULL,
                token_est INTEGER NOT NULL,
                breadcrumbs TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (doc_id, page_num, engine),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS images (
                doc_id TEXT NOT NULL,
                image_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                kind TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                bbox_json TEXT,
                phash TEXT,
                blob_relpath TEXT NOT NULL,
                caption TEXT,
                vlm_desc TEXT,
                PRIMARY KEY (doc_id, image_id),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                doc_id UNINDEXED,
                page_num UNINDEXED,
                breadcrumbs,
                content,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DocumentCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fingerprint_file(self, path: Path) -> tuple[str, int, int, bool]:
        """Return (doc_id, mtime_ns, file_size, fast_hit) using (path, mtime_ns, size) shortcut."""
        stat = path.stat()
        mtime_ns = stat.st_mtime_ns
        file_size = stat.st_size

        row = self._conn.execute(
            """
            SELECT doc_id FROM documents
            WHERE last_path = ? AND mtime_ns = ? AND file_size = ?
            LIMIT 1
            """,
            (str(path), mtime_ns, file_size),
        ).fetchone()
        if row is not None:
            return str(row["doc_id"]), mtime_ns, file_size, True

        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(65536):
                h.update(chunk)
        doc_id = h.hexdigest()[:16]
        return doc_id, mtime_ns, file_size, False

    def resolve_source(self, source: str, settings: GaviaoSettings) -> tuple[str, Path]:
        """Resolve either a cached doc_id or a filesystem path to (doc_id, Path)."""
        if re.fullmatch(r"[0-9a-f]{16}", source.strip()):
            row = self._conn.execute(
                "SELECT last_path FROM documents WHERE doc_id = ? LIMIT 1",
                (source.strip(),),
            ).fetchone()
            if row is not None:
                candidate = validate_file_path(row["last_path"], settings)
                return source.strip(), candidate

        path = validate_file_path(source, settings)
        doc_id, _, _, _ = self.fingerprint_file(path)
        return doc_id, path

    def save_document_meta(
        self,
        doc_id: str,
        path: Path,
        mtime_ns: int,
        file_size: int,
        fmt: str,
        page_count: int,
        toc_json: str,
        inspection: DocumentInspection | None = None,
    ) -> None:
        now = time.time()
        insp_json = inspection.model_dump_json() if inspection else None
        self._conn.execute(
            """
            INSERT INTO documents (
                doc_id, last_path, mtime_ns, file_size, format,
                page_count, toc_json, inspection_json, last_accessed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                last_path = excluded.last_path,
                mtime_ns = excluded.mtime_ns,
                file_size = excluded.file_size,
                format = excluded.format,
                page_count = excluded.page_count,
                toc_json = excluded.toc_json,
                inspection_json = COALESCE(excluded.inspection_json, documents.inspection_json),
                last_accessed = excluded.last_accessed
            """,
            (
                doc_id,
                str(path),
                mtime_ns,
                file_size,
                fmt,
                page_count,
                toc_json,
                insp_json,
                now,
            ),
        )
        self._conn.commit()

    def touch_document(self, doc_id: str) -> None:
        self._conn.execute(
            "UPDATE documents SET last_accessed = ? WHERE doc_id = ?",
            (time.time(), doc_id),
        )
        self._conn.commit()

    def get_inspection(self, doc_id: str) -> DocumentInspection | None:
        row = self._conn.execute(
            "SELECT inspection_json FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row and row["inspection_json"]:
            return DocumentInspection.model_validate_json(row["inspection_json"])
        return None

    def get_cached_pages(
        self,
        doc_id: str,
        page_nums: Iterable[int],
        preferred_engines: dict[int, list[str]] | None = None,
    ) -> dict[int, tuple[str, str, int, str]]:
        """Return {page_num: (engine, markdown, token_est, breadcrumbs)} for cached hits."""
        nums = list(page_nums)
        if not nums:
            return {}
        placeholders = ",".join("?" for _ in nums)
        rows = self._conn.execute(
            f"""
            SELECT page_num, engine, markdown, token_est, breadcrumbs
            FROM pages
            WHERE doc_id = ? AND page_num IN ({placeholders})
            """,
            [doc_id, *nums],
        ).fetchall()

        by_page: dict[int, list[sqlite3.Row]] = {}
        for r in rows:
            by_page.setdefault(int(r["page_num"]), []).append(r)

        result: dict[int, tuple[str, str, int, str]] = {}
        for p, candidates in by_page.items():
            allowed = preferred_engines.get(p) if preferred_engines else None
            chosen: sqlite3.Row | None = None
            if allowed:
                for eng in allowed:
                    for c in candidates:
                        if c["engine"] == eng:
                            chosen = c
                            break
                    if chosen is not None:
                        break
            else:
                chosen = candidates[0]
            if chosen is not None:
                result[p] = (
                    str(chosen["engine"]),
                    str(chosen["markdown"]),
                    int(chosen["token_est"]),
                    str(chosen["breadcrumbs"]),
                )
        return result

    def save_pages(
        self,
        doc_id: str,
        items: list[tuple[int, str, str, int, str]],
    ) -> None:
        """Persist (page_num, engine, markdown, token_est, breadcrumbs) and sync FTS5."""
        if not items:
            return
        for page_num, engine, markdown, token_est, breadcrumbs in items:
            self._conn.execute(
                """
                INSERT INTO pages (doc_id, page_num, engine, markdown, token_est, breadcrumbs)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id, page_num, engine) DO UPDATE SET
                    markdown = excluded.markdown,
                    token_est = excluded.token_est,
                    breadcrumbs = excluded.breadcrumbs
                """,
                (doc_id, page_num, engine, markdown, token_est, breadcrumbs),
            )
            self._conn.execute(
                "DELETE FROM pages_fts WHERE doc_id = ? AND page_num = ?",
                (doc_id, page_num),
            )
            self._conn.execute(
                """
                INSERT INTO pages_fts (doc_id, page_num, breadcrumbs, content)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, page_num, breadcrumbs, markdown),
            )
        self._conn.commit()
        self.touch_document(doc_id)

    def save_image_blob(
        self,
        doc_id: str,
        image_ref: ImageRef,
        webp_bytes: bytes,
        phash: str | None = None,
    ) -> None:
        doc_blob_dir = self.blobs_dir / doc_id
        doc_blob_dir.mkdir(parents=True, exist_ok=True)
        rel_path = f"{doc_id}/{image_ref.image_id}.webp"
        abs_path = self.blobs_dir / rel_path
        abs_path.write_bytes(webp_bytes)

        bbox_json = json.dumps(image_ref.bbox) if image_ref.bbox is not None else None
        self._conn.execute(
            """
            INSERT INTO images (
                doc_id, image_id, page_num, kind, width, height,
                bbox_json, phash, blob_relpath, caption, vlm_desc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, image_id) DO UPDATE SET
                width = excluded.width,
                height = excluded.height,
                bbox_json = excluded.bbox_json,
                phash = COALESCE(excluded.phash, images.phash),
                blob_relpath = excluded.blob_relpath,
                caption = COALESCE(excluded.caption, images.caption),
                vlm_desc = COALESCE(excluded.vlm_desc, images.vlm_desc)
            """,
            (
                doc_id,
                image_ref.image_id,
                image_ref.page,
                image_ref.kind,
                image_ref.width,
                image_ref.height,
                bbox_json,
                phash,
                rel_path,
                image_ref.caption,
                image_ref.vlm_description,
            ),
        )
        self._conn.commit()

    def update_image_vlm_desc(self, doc_id: str, image_id: str, desc: str) -> None:
        self._conn.execute(
            "UPDATE images SET vlm_desc = ? WHERE doc_id = ? AND image_id = ?",
            (desc, doc_id, image_id),
        )
        self._conn.commit()

    def get_page_images(self, doc_id: str, page_nums: Iterable[int]) -> list[ImageRef]:
        nums = list(page_nums)
        if not nums:
            return []
        placeholders = ",".join("?" for _ in nums)
        rows = self._conn.execute(
            f"""
            SELECT image_id, page_num, kind, width, height, bbox_json, caption, vlm_desc
            FROM images
            WHERE doc_id = ? AND page_num IN ({placeholders})
            ORDER BY page_num, image_id
            """,
            [doc_id, *nums],
        ).fetchall()
        return [self._row_to_image_ref(doc_id, r) for r in rows]

    def get_image(self, doc_id: str, image_id: str) -> tuple[ImageRef, bytes] | None:
        row = self._conn.execute(
            """
            SELECT image_id, page_num, kind, width, height, bbox_json, blob_relpath, caption, vlm_desc
            FROM images
            WHERE doc_id = ? AND image_id = ?
            LIMIT 1
            """,
            (doc_id, image_id),
        ).fetchone()
        if row is None:
            return None
        blob_path = self.blobs_dir / str(row["blob_relpath"])
        if not blob_path.exists():
            return None
        return self._row_to_image_ref(doc_id, row), blob_path.read_bytes()

    @staticmethod
    def _row_to_image_ref(doc_id: str, row: sqlite3.Row) -> ImageRef:
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
        image_id = str(row["image_id"])
        return ImageRef(
            image_id=image_id,
            page=int(row["page_num"]),
            kind=str(row["kind"]),
            width=int(row["width"]),
            height=int(row["height"]),
            uri=f"doc://{doc_id}/img/{image_id}",
            bbox=bbox,
            caption=row["caption"],
            vlm_description=row["vlm_desc"],
        )

    def get_indexed_page_nums(self, doc_id: str) -> set[int]:
        rows = self._conn.execute(
            "SELECT DISTINCT page_num FROM pages_fts WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
        return {int(r["page_num"]) for r in rows}

    def search_fts(self, doc_id: str, query: str, limit: int = 10) -> list[SearchHit]:
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not tokens:
            return []
        fts_query = " ".join(f'"{t}"' for t in tokens)
        rows = self._conn.execute(
            """
            SELECT page_num, breadcrumbs,
                   bm25(pages_fts) AS rank_score,
                   snippet(pages_fts, 3, '«', '»', '…', 24) AS snip
            FROM pages_fts
            WHERE pages_fts MATCH ? AND doc_id = ?
            ORDER BY rank_score
            LIMIT ?
            """,
            (fts_query, doc_id, limit),
        ).fetchall()
        return [
            SearchHit(
                page=int(r["page_num"]),
                score=round(abs(float(r["rank_score"])), 4),
                breadcrumbs=str(r["breadcrumbs"] or ""),
                snippet=str(r["snip"] or ""),
            )
            for r in rows
        ]

    def search_regex(self, doc_id: str, pattern: str, limit: int = 10) -> list[SearchHit]:
        compiled = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)
        rows = self._conn.execute(
            """
            SELECT page_num, breadcrumbs, markdown
            FROM pages
            WHERE doc_id = ?
            ORDER BY page_num
            """,
            (doc_id,),
        ).fetchall()

        seen_pages: set[int] = set()
        hits: list[SearchHit] = []
        for r in rows:
            p = int(r["page_num"])
            if p in seen_pages:
                continue
            text = str(r["markdown"])
            matches = list(compiled.finditer(text))
            if matches:
                seen_pages.add(p)
                m = matches[0]
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                snippet = text[start:end].replace("\n", " ").strip()
                if start > 0:
                    snippet = "…" + snippet
                if end < len(text):
                    snippet = snippet + "…"
                hits.append(
                    SearchHit(
                        page=p,
                        score=float(len(matches)),
                        breadcrumbs=str(r["breadcrumbs"] or ""),
                        snippet=snippet,
                    )
                )
                if len(hits) >= limit:
                    break
        return hits

    def total_size_bytes(self) -> int:
        total = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    def enforce_lru_budget(self) -> int:
        """Evict least-recently-accessed documents if cache exceeds max_cache_bytes."""
        evicted = 0
        while self.total_size_bytes() > self.max_cache_bytes:
            oldest = self._conn.execute(
                "SELECT doc_id FROM documents ORDER BY last_accessed ASC LIMIT 1"
            ).fetchone()
            if oldest is None:
                break
            victim = str(oldest["doc_id"])
            self._conn.execute("DELETE FROM pages_fts WHERE doc_id = ?", (victim,))
            self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (victim,))
            self._conn.commit()
            blob_dir = self.blobs_dir / victim
            if blob_dir.exists():
                shutil.rmtree(blob_dir, ignore_errors=True)
            evicted += 1
        return evicted

    def stats(self) -> dict[str, int | str]:
        doc_count = int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        page_count = int(self._conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
        img_count = int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        return {
            "cache_dir": str(self.cache_dir),
            "documents": doc_count,
            "cached_pages": page_count,
            "cached_images": img_count,
            "total_bytes": self.total_size_bytes(),
            "max_bytes": self.max_cache_bytes,
        }

    def clear(self) -> int:
        doc_count = int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        self._conn.execute("DELETE FROM pages_fts")
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()
        if self.blobs_dir.exists():
            shutil.rmtree(self.blobs_dir, ignore_errors=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        return doc_count
