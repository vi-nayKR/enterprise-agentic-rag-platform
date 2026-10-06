import re
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.models import Document, DocumentChunk


class SemanticChunker:
    """Source-preserving fixed, recursive, or embedding-boundary splitting.

    The historical class name is retained for callers; recursive remains default.
    Semantic mode requires the async method and an explicitly supplied embedder.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150,
                 mode: Literal["fixed", "recursive", "semantic"] = "recursive"):
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require positive chunk size and overlap smaller than the chunk")
        if mode not in {"fixed", "recursive", "semantic"}:
            raise ValueError("Unknown chunking mode")
        self.chunk_size, self.chunk_overlap, self.mode = chunk_size, chunk_overlap, mode

    def _chunks(self, document: Document, offsets: list[tuple[int, int]]) -> list[DocumentChunk]:
        chunks = []
        for index, (start, end) in enumerate(offsets):
            text = document.text[start:end]
            header = re.search(r"^#+\s+(.+)$", text, re.MULTILINE)
            chunks.append(DocumentChunk(
                id=f"{document.id}_{self.mode}_{index}", document_id=document.id,
                chunk_index=index, text=text, token_count=max(1, len(text) // 4),
                metadata={**document.metadata, "filename": document.filename,
                          "section": header.group(1) if header else document.metadata.get("title", "General"),
                          "total_chunks": len(offsets), "start_offset": start, "end_offset": end,
                          "chunking": self.mode},
            ))
        return chunks

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        if not document.text:
            return []
        if self.mode == "semantic":
            raise ValueError("Semantic mode requires chunk_document_async(document, embeddings)")
        if self.mode == "fixed":
            offsets = []
            start = 0
            while start < len(document.text):
                end = min(len(document.text), start + self.chunk_size)
                offsets.append((start, end))
                if end == len(document.text):
                    break
                start = end - self.chunk_overlap
            return self._chunks(document, offsets)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True, strip_whitespace=False,
        )
        offsets = []
        for chunk in splitter.create_documents([document.text]):
            start = chunk.metadata["start_index"]
            end = start + len(chunk.page_content)
            if start < 0 or document.text[start:end] != chunk.page_content:
                raise ValueError("Recursive splitter did not preserve a verifiable source offset")
            offsets.append((start, end))
        return self._chunks(document, offsets)

    async def chunk_document_async(self, document: Document, embeddings) -> list[DocumentChunk]:
        if self.mode != "semantic":
            return self.chunk_document(document)
        if not document.text:
            return []
        ends = [match.end() for match in re.finditer(r"(?<=[.!?])\s+|\n+", document.text)]
        ends.append(len(document.text))
        starts = [0] + ends[:-1]
        sentences = [document.text[start:end] for start, end in zip(starts, ends)]
        vectors = await embeddings.embed_documents(sentences)
        if len(vectors) != len(sentences):
            raise ValueError("Sentence embedding count does not match source sentences")
        boundaries = [ends[index] for index in range(len(vectors) - 1)
                      if sum(a * b for a, b in zip(vectors[index], vectors[index + 1])) < 0.5]
        # ponytail: adjacent-sentence cosine with a fixed threshold; tune on dev
        # or add buffered context if measured boundary quality warrants it.
        offsets, start = [], 0
        while start < len(document.text):
            limit = min(len(document.text), start + self.chunk_size)
            minimum = start + max(self.chunk_overlap + 1, self.chunk_size // 4)
            choices = [point for point in boundaries if minimum <= point <= limit]
            end = choices[0] if choices else limit
            offsets.append((start, end))
            if end == len(document.text):
                break
            start = end - self.chunk_overlap
        return self._chunks(document, offsets)
