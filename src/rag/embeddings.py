import hashlib
import math
from typing import Any, List, cast

from config.settings import settings

try:
    from langchain_openai import OpenAIEmbeddings

    HAS_OPENAI = True
except ImportError:
    OpenAIEmbeddings = None
    HAS_OPENAI = False


class EmbeddingsService:
    """
    Provides dense vector embeddings.
    Uses OpenAI Embeddings when an API key is present; otherwise falls back to
    deterministic offline normalized vectors for local testing.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.use_openai = bool(
            HAS_OPENAI
            and settings.OPENAI_API_KEY is not None
            and not settings.OPENAI_API_KEY.startswith("sk-placeholder")
            and not settings.OPENAI_API_KEY.startswith("sk-mock")
        )
        self._client: Any = None
        if self.use_openai and OpenAIEmbeddings is not None:
            try:
                kwargs: dict[str, Any] = {
                    "model": settings.EMBEDDING_MODEL,
                    "openai_api_key": settings.OPENAI_API_KEY,
                }
                if settings.OPENAI_BASE_URL:
                    kwargs["base_url"] = settings.OPENAI_BASE_URL
                self._client = OpenAIEmbeddings(**cast(dict[str, Any], kwargs))
            except Exception:
                self.use_openai = False
                self._client = None

    def _generate_token_vector(self, token: str) -> List[float]:
        """Generates a deterministic 1536-dim unit vector for a single token."""
        vec = []
        for i in range(self.dimension):
            hash_input = f"{token}:{i}".encode("utf-8")
            hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
            val = (hash_val / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)
        return vec

    def _generate_offline_embedding(self, text: str) -> List[float]:
        """
        Generates a deterministic semantic embedding via token composition.
        Texts sharing semantic keywords will have high cosine similarity.
        """
        import re
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0] * self.dimension

        # Sum token vectors (bag-of-words semantic composition)
        composite_vector = [0.0] * self.dimension
        for token in tokens:
            tok_vec = self._generate_token_vector(token)
            for i in range(self.dimension):
                composite_vector[i] += tok_vec[i]

        # L2-normalize composite vector
        norm = math.sqrt(sum(x * x for x in composite_vector))
        return [x / norm for x in composite_vector] if norm > 0 else composite_vector

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text chunks."""
        if self.use_openai:
            try:
                return await self._client.aembed_documents(texts)
            except Exception as e:
                print(
                    f"[Embeddings] OpenAI failed ({e}), falling back to offline generator."
                )
        return [self._generate_offline_embedding(t) for t in texts]

    async def embed_query(self, text: str) -> List[float]:
        """Generates an embedding for a single user query."""
        if self.use_openai:
            try:
                return await self._client.aembed_query(text)
            except Exception as e:
                print(
                    f"[Embeddings] OpenAI failed ({e}), falling back to offline generator."
                )
        return self._generate_offline_embedding(text)
