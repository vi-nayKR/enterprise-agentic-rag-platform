import time
from collections import OrderedDict
from typing import List, Optional, Any, Dict
from src.rag.models import SearchResult

class SemanticQueryCache:
    """
    In-memory LRU cache with TTL for query embeddings and hybrid search results.
    Provides sub-10ms retrieval latency for frequent and repeated queries.
    """
    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        # OrderedDict for LRU: key -> (value, expiry_timestamp)
        self.results_cache: OrderedDict[str, tuple[List[SearchResult], float]] = OrderedDict()
        self.embedding_cache: OrderedDict[str, tuple[List[float], float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _normalize_key(self, text: str) -> str:
        return text.strip().lower()

    def get_results(self, query: str) -> Optional[List[SearchResult]]:
        """Retrieves cached search results if present and not expired."""
        key = self._normalize_key(query)
        if key in self.results_cache:
            results, expiry = self.results_cache[key]
            if time.time() < expiry:
                self.results_cache.move_to_end(key)
                self.hits += 1
                return results
            else:
                del self.results_cache[key]
        self.misses += 1
        return None

    def set_results(self, query: str, results: List[SearchResult], ttl: Optional[int] = None):
        """Stores search results in LRU cache."""
        key = self._normalize_key(query)
        expiry = time.time() + (ttl or self.default_ttl)
        if key in self.results_cache:
            self.results_cache.move_to_end(key)
        self.results_cache[key] = (results, expiry)
        if len(self.results_cache) > self.max_size:
            self.results_cache.popitem(last=False)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieves cached embedding vector."""
        key = self._normalize_key(text)
        if key in self.embedding_cache:
            vec, expiry = self.embedding_cache[key]
            if time.time() < expiry:
                self.embedding_cache.move_to_end(key)
                return vec
            else:
                del self.embedding_cache[key]
        return None

    def set_embedding(self, text: str, vector: List[float], ttl: Optional[int] = None):
        """Stores embedding vector in LRU cache."""
        key = self._normalize_key(text)
        expiry = time.time() + (ttl or self.default_ttl)
        if key in self.embedding_cache:
            self.embedding_cache.move_to_end(key)
        self.embedding_cache[key] = (vector, expiry)
        if len(self.embedding_cache) > self.max_size:
            self.embedding_cache.popitem(last=False)

    def clear(self):
        """Clears all caches."""
        self.results_cache.clear()
        self.embedding_cache.clear()
        self.hits = 0
        self.misses = 0

query_cache = SemanticQueryCache()
