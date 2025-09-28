import os
import shutil
import tempfile
import logging
import contextlib
from collections import OrderedDict
from typing import Any
from uuid import UUID

from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher

from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest
from api.vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
    FilterArg,
    FilterCondition,
    FilterGroup,
    FilterOperator,
)


class LuceneVectorStoreRepository(VectorStoreRepository):
    """
    VectorStoreRepository backed entirely by Lucene/Pyserini (BM25).
    - Index via LuceneIndexer
    - Search via LuceneSearcher (BM25)
    - One Lucene doc per MemoryRequest (field: 'contents')
    - No chunking; no embedding model needed
    """

    def __init__(
        self,
        index_dir: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        overwrite: bool = False,
    ):
        super().__init__(embedding_model=None, text_chunker=None)
        self._k1 = k1
        self._b = b

        # Prepare index directory
        self._index_dir = index_dir or tempfile.mkdtemp(prefix="lucene_memories_")
        if overwrite and os.path.exists(self._index_dir):
            shutil.rmtree(self._index_dir)
        os.makedirs(self._index_dir, exist_ok=True)

        # If a Lucene index already exists (segments_* present), we append to it.
        has_segments = any(
            fn.startswith("segments_") for fn in os.listdir(self._index_dir)
        )
        indexer_args = ["-index", self._index_dir, "-storeContents", "-storeDocvectors"]
        self._indexer: LuceneIndexer | None = LuceneIndexer(
            index_dir=self._index_dir, args=indexer_args, append=has_segments
        )
        self._pending_writes: bool = False  # track uncommitted docs

        self._memories: OrderedDict[UUID, MemoryRequest] = OrderedDict()

        logging.info(
            f"LuceneVectorStoreRepository ready at {self._index_dir} "
            f"(BM25 k1={self._k1}, b={self._b}, append={has_segments})"
        )

    # ---------------- Context manager & destructor ----------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._commit_and_close_indexer()
        finally:
            # Nothing else to release; LuceneSearcher is created on demand per search
            pass

    def __del__(self):
        # Be defensive: finalize writes if user forgot to call finalize_index()
        with contextlib.suppress(Exception):
            self._commit_and_close_indexer()

    # ---------------- Core API ----------------

    async def index_memory(self, memory: MemoryRequest) -> None:
        """
        Add or overwrite a memory in the Lucene index and local registry.
        """
        # (Re)open indexer in append mode if it was closed
        # (e.g., after finalize or a previous search)
        if self._indexer is None:
            self._indexer = LuceneIndexer(
                index_dir=self._index_dir,
                args=["-index", self._index_dir, "-storeContents", "-storeDocvectors"],
                append=True,
            )

        text = " ".join(memory.text)
        self._indexer.add_doc_dict({"id": str(memory.id), "contents": text})
        self._pending_writes = True

        self._memories[memory.id] = memory
        logging.debug(f"[Lucene] Indexed memory {memory.id}")

    async def get_memory(self, memory_id: UUID) -> MemoryRequest | None:
        return self._memories.get(memory_id)

    async def search_similar(
        self,
        query: MemoryRequest,
        limit: int = 5,
        filters: FilterArg = None,
    ) -> MemoryContext:
        """
        BM25 search via LuceneSearcher over the 'contents' field.
        """
        qtext = " ".join(query.text).strip()
        context = MemoryContext.create(query)
        if not qtext:
            return context

        # Ensure any pending writes are committed so segments_* exist
        await self._ensure_committed()

        # Create a fresh searcher (cheap; ensures it sees latest commits)
        searcher = LuceneSearcher(self._index_dir)
        searcher.set_bm25(k1=self._k1, b=self._b)

        # Over-retrieve to allow Python-side filters; then trim to 'limit'
        k = max(limit * 4, limit)
        hits = searcher.search(qtext, k=k)

        for h in hits:
            # You indexed with {'id': str(UUID)}, so docid is the UUID string
            try:
                mem_id = UUID(h.docid)
            except Exception:
                continue

            mem = self._memories.get(mem_id)
            if mem is None:
                continue

            if not self._passes_filters(mem, filters):
                continue

            context.add_memory(
                memory=mem,
                score=float(h.score),
                matched_text=" ".join(mem.text),
            )

            if len(context.memories) >= limit:
                break

        return context

    async def delete_memory(self, memory_id: UUID) -> None:
        """
        Simple/robust delete: remove from registry and rebuild the index.
        """
        if memory_id in self._memories:
            del self._memories[memory_id]
        self._rebuild_index_from_registry()

    async def list_memories(
        self, limit: int = 100, offset: int = 0, filters: FilterArg = None
    ) -> tuple[list[MemoryRequest], UUID | None]:
        """
        Offset-based pagination over in-memory registry. The second return
        value is a UUID cursor in your interface; here we return None.
        """
        items = [m for m in self._memories.values() if self._passes_filters(m, filters)]
        page = items[offset : offset + limit]
        next_cursor = None  # not using UUID cursoring here
        return page, next_cursor

    # ---------------- Public helper ----------------

    async def finalize_index(self) -> None:
        """
        Commit pending writes so a segments_N exists and searchers can open.
        Call this at the end of ingestion (e.g., in your DatasetLoader).
        """
        await self._ensure_committed()

    # ---------------- Internals ----------------

    async def _ensure_committed(self) -> None:
        """
        If there are pending writes, commit them (close the indexer).
        Also ensures future indexing reopens the indexer in append mode.
        """
        if self._pending_writes:
            self._commit_and_close_indexer()
            self._pending_writes = False

    def _commit_and_close_indexer(self) -> None:
        if self._indexer is not None:
            try:
                self._indexer.close()  # commits segments_*
            finally:
                self._indexer = None

    def _rebuild_index_from_registry(self) -> None:
        """
        Close any open indexer, wipe the dir, and rebuild from current registry.
        """
        self._commit_and_close_indexer()

        # wipe dir
        if os.path.exists(self._index_dir):
            shutil.rmtree(self._index_dir)
        os.makedirs(self._index_dir, exist_ok=True)

        # fresh indexer (create mode)
        self._indexer = LuceneIndexer(
            index_dir=self._index_dir,
            args=["-index", self._index_dir, "-storeContents", "-storeDocvectors"],
            append=False,
        )
        for mem in self._memories.values():
            self._indexer.add_doc_dict({
                "id": str(mem.id),
                "contents": " ".join(mem.text),
            })
            self._pending_writes = True

        # commit rebuild
        self._commit_and_close_indexer()
        self._pending_writes = False

    # ---------------- Filters ----------------

    def _passes_filters(self, mem: MemoryRequest, filters: FilterArg) -> bool:
        if filters is None:
            return True
        if isinstance(filters, FilterCondition):
            return self._eval_condition(mem, filters)
        if isinstance(filters, FilterGroup):
            children = [self._passes_filters(mem, c) for c in filters.conditions]
            return all(children) if filters.operator.upper() == "AND" else any(children)
        return True

    def _eval_condition(self, mem: MemoryRequest, cond: FilterCondition) -> bool:
        val = self._get_field(mem, cond.field)
        op = cond.operator
        cmpv = cond.value
        if op == FilterOperator.EXISTS:
            return val is not None
        if op == FilterOperator.NOT_EXISTS:
            return val is None
        if op == FilterOperator.EQUALS:
            return val == cmpv
        if op == FilterOperator.NOT_EQUALS:
            return val != cmpv
        if op == FilterOperator.GREATER_THAN:
            return (val is not None) and (cmpv is not None) and val > cmpv
        if op == FilterOperator.GREATER_THAN_OR_EQUAL:
            return (val is not None) and (cmpv is not None) and val >= cmpv
        if op == FilterOperator.LESS_THAN:
            return (val is not None) and (cmpv is not None) and val < cmpv
        if op == FilterOperator.LESS_THAN_OR_EQUAL:
            return (val is not None) and (cmpv is not None) and val <= cmpv
        if op == FilterOperator.CONTAINS:
            try:
                if isinstance(val, list):
                    return cmpv in val
                if isinstance(val, str):
                    return str(cmpv) in val
            except Exception:
                return False
            return False
        return True

    def _get_field(self, mem: MemoryRequest, dotted: str) -> Any:
        # Your MemoryRequest appears to expose simple top-level attributes.
        # If you later nest metadata, expand this resolver accordingly.
        try:
            return getattr(mem, dotted)
        except AttributeError:
            return None
