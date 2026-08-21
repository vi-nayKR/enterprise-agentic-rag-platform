import re
from typing import List
from src.rag.models import Document, DocumentChunk

class SemanticChunker:
    """
    Splits text recursively based on structural markdown and natural language boundaries
    while maintaining target chunk size and overlap.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = [
            "\n\n# ",    # H1 markdown headers
            "\n\n## ",   # H2 markdown headers
            "\n\n### ",  # H3 markdown headers
            "\n\n",      # Paragraphs
            "\n",        # Line breaks
            ". ",        # Sentences
            " "          # Words
        ]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using the hierarchy of separators."""
        final_chunks: List[str] = []
        if not separators:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator) if separator in text else [text]

        accumulated_chunk = ""
        for piece in splits:
            candidate = f"{accumulated_chunk}{separator}{piece}".strip() if accumulated_chunk else piece.strip()

            if len(candidate) <= self.chunk_size:
                accumulated_chunk = candidate
            else:
                if accumulated_chunk:
                    final_chunks.append(accumulated_chunk)
                    # Slide overlap
                    overlap_start = max(0, len(accumulated_chunk) - self.chunk_overlap)
                    accumulated_chunk = accumulated_chunk[overlap_start:]
                    candidate = f"{accumulated_chunk} {piece}".strip()

                if len(piece) > self.chunk_size and remaining_separators:
                    sub_chunks = self._split_text_recursive(piece, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    accumulated_chunk = ""
                else:
                    accumulated_chunk = piece.strip()

        if accumulated_chunk and accumulated_chunk not in final_chunks:
            final_chunks.append(accumulated_chunk)

        return [c for c in final_chunks if len(c.strip()) > 0]

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Chunks a Document into a list of enriched DocumentChunk objects."""
        raw_chunks = self._split_text_recursive(document.text, self.separators)
        chunks: List[DocumentChunk] = []

        for idx, text in enumerate(raw_chunks):
            # Estimate token count (~4 characters per token)
            token_count = max(1, len(text) // 4)

            # Extract header context if present
            header_match = re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)
            section_header = header_match[0] if header_match else document.metadata.get("title", "General")

            chunk_metadata = {
                **document.metadata,
                "filename": document.filename,
                "section": section_header,
                "total_chunks": len(raw_chunks)
            }

            chunks.append(
                DocumentChunk(
                    id=f"{document.id}_chunk_{idx}",
                    document_id=document.id,
                    chunk_index=idx,
                    text=text,
                    token_count=token_count,
                    metadata=chunk_metadata
                )
            )
        return chunks
