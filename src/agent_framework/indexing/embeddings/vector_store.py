"""LanceDB-backed vector store for embedded code symbols."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent_framework.indexing.embeddings.embedder import Embedder
    from agent_framework.indexing.models import CodebaseIndex

logger = logging.getLogger(__name__)


def _symbol_doc_text(sym) -> str:
    """Build search_document text for a symbol."""
    parts = [f"{sym.kind} {sym.name} in {sym.file_path}"]
    if sym.signature:
        parts.append(sym.signature)
    if sym.docstring:
        parts.append(sym.docstring)
    return "\n".join(parts)


def _module_doc_text(mod) -> str:
    """Build search_document text for a module."""
    parts = [f"module {mod.path}/ ({mod.file_count} files, {mod.language})"]
    if mod.description:
        parts.append(mod.description)
    if mod.key_files:
        parts.append(f"key files: {', '.join(mod.key_files)}")
    return "\n".join(parts)


_TABLE_NAME = "embeddings"


class VectorStore:
    """Wraps LanceDB for vector similarity search over code symbols."""

    def __init__(self, store_path: Path, dimensions: int = 256) -> None:
        self._store_path = store_path
        self._dimensions = dimensions
        self._db = None

    def _ensure_db(self):
        if self._db is None:
            import lancedb
            self._store_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self._store_path))
        return self._db

    def _has_table(self) -> bool:
        db = self._ensure_db()
        return _TABLE_NAME in db.table_names()

    def is_stale(self, commit_sha: str) -> bool:
        """Check if stored embeddings match the given commit SHA."""
        if not self._has_table():
            return True
        try:
            tbl = self._ensure_db().open_table(_TABLE_NAME)
            # Peek at first row's commit_sha
            rows = tbl.head(1).to_pydict()
            if not rows or not rows.get("commit_sha"):
                return True
            stored = rows["commit_sha"][0]
            return stored != commit_sha
        except Exception:
            return True

    def rebuild(self, index: CodebaseIndex, embedder: Embedder) -> None:
        """Drop and recreate the entire embedding table."""
        db = self._ensure_db()

        # Build documents from symbols + modules
        docs, metas = self._prepare_documents(index)
        if not docs:
            logger.info("No documents to embed for %s", index.repo_slug)
            return

        vectors = embedder.embed_texts(docs)
        if vectors is None:
            logger.warning("Embedding failed during rebuild for %s", index.repo_slug)
            return

        # Drop existing table
        if _TABLE_NAME in db.table_names():
            db.drop_table(_TABLE_NAME)

        records = self._build_records(metas, vectors, index.commit_sha)
        schema = self._make_schema()
        tbl = db.create_table(_TABLE_NAME, schema=schema)
        tbl.add(records)
        logger.info("Embedded %d symbols + modules for %s", len(records), index.repo_slug)

    def update_incremental(
        self,
        index: CodebaseIndex,
        embedder: Embedder,
        changed_files: set[str],
        deleted_files: set[str],
    ) -> None:
        """Re-embed only changed files, delete removed ones."""
        if not self._has_table():
            self.rebuild(index, embedder)
            return

        db = self._ensure_db()
        tbl = db.open_table(_TABLE_NAME)

        # Delete rows matching changed or deleted file paths
        remove_paths = changed_files | deleted_files
        if remove_paths:
            # Escape single quotes to prevent malformed filter expressions
            filter_expr = " OR ".join(
                f"file_path = '{p.replace(chr(39), chr(39)+chr(39))}'"
                for p in sorted(remove_paths)
            )
            try:
                tbl.delete(filter_expr)
            except Exception:
                logger.debug("Delete during incremental update failed", exc_info=True)

        # Re-embed symbols from changed files
        changed_syms = [s for s in index.symbols if s.file_path in changed_files]
        # Modules whose path matches a changed file's directory
        changed_dirs = {str(Path(f).parent) for f in changed_files}
        changed_mods = [m for m in index.modules if m.path in changed_dirs]

        if not changed_syms and not changed_mods:
            # Just update commit_sha on existing rows
            self._update_commit_sha(tbl, index.commit_sha)
            return

        docs = []
        metas = []
        for sym in changed_syms:
            docs.append(_symbol_doc_text(sym))
            metas.append({
                "id": f"sym:{sym.file_path}:{sym.line}:{sym.name}",
                "kind": str(sym.kind),
                "name": sym.name,
                "file_path": sym.file_path,
                "line": sym.line,
                "signature": sym.signature or "",
                "docstring": sym.docstring or "",
                "parent": sym.parent or "",
            })
        for mod in changed_mods:
            docs.append(_module_doc_text(mod))
            metas.append({
                "id": f"mod:{mod.path}",
                "kind": "module",
                "name": mod.path,
                "file_path": mod.path,
                "line": 0,
                "signature": "",
                "docstring": mod.description,
                "parent": "",
            })

        vectors = embedder.embed_texts(docs)
        if vectors is None:
            logger.warning("Embedding failed during incremental update")
            return

        records = self._build_records(metas, vectors, index.commit_sha)
        tbl.add(records)
        logger.info(
            "Incremental update: %d changed, %d deleted for %s",
            len(records), len(deleted_files), index.repo_slug,
        )

    def query(self, embedding: list[float], n_results: int = 15) -> list[dict]:
        """Vector similarity search, returns metadata dicts."""
        if not self._has_table():
            return []
        try:
            tbl = self._ensure_db().open_table(_TABLE_NAME)
            results = (
                tbl.search(embedding)
                .limit(n_results)
                .to_pandas()
            )
            rows = []
            for _, row in results.iterrows():
                rows.append({
                    "id": row.get("id", ""),
                    "kind": row.get("kind", ""),
                    "name": row.get("name", ""),
                    "file_path": row.get("file_path", ""),
                    "line": int(row.get("line", 0)),
                    "signature": row.get("signature", ""),
                    "docstring": row.get("docstring", ""),
                    "parent": row.get("parent", ""),
                    "_distance": float(row.get("_distance", 0.0)),
                })
            return rows
        except Exception:
            logger.warning("Vector query failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_documents(self, index: CodebaseIndex):
        """Build parallel lists of doc texts and metadata dicts."""
        docs = []
        metas = []
        for sym in index.symbols:
            docs.append(_symbol_doc_text(sym))
            metas.append({
                "id": f"sym:{sym.file_path}:{sym.line}:{sym.name}",
                "kind": str(sym.kind),
                "name": sym.name,
                "file_path": sym.file_path,
                "line": sym.line,
                "signature": sym.signature or "",
                "docstring": sym.docstring or "",
                "parent": sym.parent or "",
            })
        for mod in index.modules:
            docs.append(_module_doc_text(mod))
            metas.append({
                "id": f"mod:{mod.path}",
                "kind": "module",
                "name": mod.path,
                "file_path": mod.path,
                "line": 0,
                "signature": "",
                "docstring": mod.description,
                "parent": "",
            })
        return docs, metas

    @staticmethod
    def _build_records(metas, vectors, commit_sha):
        records = []
        for meta, vec in zip(metas, vectors):
            record = dict(meta)
            record["vector"] = vec
            record["text"] = ""
            record["commit_sha"] = commit_sha
            records.append(record)
        return records

    def _make_schema(self):
        import pyarrow as pa
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self._dimensions)),
            pa.field("kind", pa.string()),
            pa.field("name", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("line", pa.int64()),
            pa.field("signature", pa.string()),
            pa.field("docstring", pa.string()),
            pa.field("parent", pa.string()),
            pa.field("commit_sha", pa.string()),
        ])

    @staticmethod
    def _update_commit_sha(tbl, commit_sha: str) -> None:
        """No-op: LanceDB lacks cheap in-place column updates.

        Stale SHA is accepted until next rebuild when symbols actually change.
        """
