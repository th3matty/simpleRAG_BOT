from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time
import re

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """
    Represents a processed document chunk with its metadata.
    This structured format ensures consistent handling of document pieces.
    """

    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    similarity_score: float = 0.0
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None

    def __post_init__(self):
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []


class DocumentProcessor:
    """
    Handles document processing with an adaptive chunking strategy.
    Adjusts chunk sizes based on content structure while maintaining efficiency.
    """

    # Chunk size limits
    MAX_CHUNK_SIZE = 1200  # Maximum characters per chunk
    MIN_CHUNK_SIZE = 400  # Minimum characters per chunk
    OVERLAP_SIZE = 200  # Overlap size in characters

    # Content structure patterns
    HEADER_PATTERNS = [
        r"^#{1,6}\s+.+$",  # Markdown headers
        r"^[A-Z][^\n]+\n[=\-]{2,}$",  # Underlined headers
    ]

    LIST_PATTERNS = [
        r"^\s*[-*+]\s+.+$",  # Unordered lists
        r"^\s*\d+\.\s+.+$",  # Ordered lists
    ]

    CODE_BLOCK_PATTERNS = [
        r"```[\s\S]*?```",  # Fenced code blocks
        r"(?:(?:^|\n)\s{4}[^\n]+)+",  # Indented code blocks
    ]

    def __init__(
        self, embedding_service, max_chunk_size: int = 800, chunk_overlap: int = 200
    ):
        """
        Initialize the document processor with necessary parameters.

        Args:
            embedding_service: Service for generating text embeddings
            max_chunk_size: Maximum size of text chunks (in characters)
        """
        self.embedding_service = embedding_service
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized DocumentProcessor with max_chunk_size: {max_chunk_size}"
        )

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a piece of text."""
        return self.embedding_service.get_single_embedding(text)

    def _identify_content_structure(self, text: str) -> Dict[str, List[tuple]]:
        """
        Identify different content structures in the text and their positions.

        Returns:
            Dictionary mapping structure types to lists of (start, end) positions
        """
        structures = {"headers": [], "lists": [], "code_blocks": [], "paragraphs": []}

        # Find headers
        for pattern in self.HEADER_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                structures["headers"].append((match.start(), match.end()))

        # Find lists
        list_items = []
        for pattern in self.LIST_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                list_items.append((match.start(), match.end()))
        # Group adjacent list items
        if list_items:
            list_start = list_items[0][0]
            prev_end = list_items[0][1]
            for start, end in list_items[1:]:
                if start - prev_end > 2:  # New list if gap > 2 lines
                    structures["lists"].append((list_start, prev_end))
                    list_start = start
                prev_end = end
            structures["lists"].append((list_start, prev_end))

        # Find code blocks
        for pattern in self.CODE_BLOCK_PATTERNS:
            for match in re.finditer(pattern, text):
                structures["code_blocks"].append((match.start(), match.end()))

        # Find paragraphs (text between blank lines, excluding other structures)
        paragraph_pattern = r"(?:^|\n\n)((?:[^\n]+\n?)+)(?=\n\n|$)"
        for match in re.finditer(paragraph_pattern, text):
            # Check if this region overlaps with other structures
            start, end = match.start(1), match.end(1)
            is_special = False
            for struct_type in ["headers", "lists", "code_blocks"]:
                for s_start, s_end in structures[struct_type]:
                    if start <= s_end and end >= s_start:
                        is_special = True
                        break
                if is_special:
                    break
            if not is_special:
                structures["paragraphs"].append((start, end))

        return structures

    def _split_into_sections(self, text: str) -> List[str]:
        """
        Split text into sections by natural breaks, respecting content structure.
        """
        # Identify all content structures
        structures = self._identify_content_structure(text)

        # Combine all break points
        break_points = []
        for struct_type, positions in structures.items():
            for start, end in positions:
                break_points.extend(
                    [(start, "start", struct_type), (end, "end", struct_type)]
                )

        # Sort break points by position
        break_points.sort(key=lambda x: x[0])

        # Create sections based on structure boundaries
        sections = []
        current_start = 0

        for pos, point_type, struct_type in break_points:
            if point_type == "start":
                # If there's text before this structure, add it as a section
                if pos > current_start:
                    section_text = text[current_start:pos].strip()
                    if section_text:
                        sections.append(section_text)
                # Add the structure as its own section
                next_end = next(
                    (
                        p[0]
                        for p in break_points
                        if p[0] > pos and p[1] == "end" and p[2] == struct_type
                    ),
                    len(text),
                )
                section_text = text[pos:next_end].strip()
                if section_text:
                    sections.append(section_text)
                current_start = next_end

        # Add any remaining text
        if current_start < len(text):
            section_text = text[current_start:].strip()
            if section_text:
                sections.append(section_text)

        logger.debug(f"Split document into {len(sections)} sections")
        return sections

    def _estimate_chunk_size(self, section: str) -> int:
        """
        Estimate appropriate chunk size based on content structure.
        Returns a size between MIN_CHUNK_SIZE and MAX_CHUNK_SIZE.
        """
        # Check for structural indicators
        has_headers = any(
            re.search(pattern, section, re.MULTILINE)
            for pattern in self.HEADER_PATTERNS
        )
        has_lists = any(
            re.search(pattern, section, re.MULTILINE) for pattern in self.LIST_PATTERNS
        )
        has_code = any(
            re.search(pattern, section) for pattern in self.CODE_BLOCK_PATTERNS
        )

        # Start with base size
        size = (self.MAX_CHUNK_SIZE + self.MIN_CHUNK_SIZE) // 2

        # Adjust based on content structure
        if has_headers:
            size = max(
                size - 200, self.MIN_CHUNK_SIZE
            )  # Smaller chunks for structured content
        if has_lists or has_code:
            size = max(
                size - 100, self.MIN_CHUNK_SIZE
            )  # Slightly smaller for lists/code
        if not (has_headers or has_lists or has_code):
            size = min(size + 200, self.MAX_CHUNK_SIZE)  # Larger for prose

        return size

    def _split_into_semantic_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks with adaptive sizing based on content structure.
        Uses lightweight heuristics for better performance.
        """
        sections = self._split_into_sections(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Quick check for natural break points
            has_break = bool(re.search(r"\n\n|\.\s+[A-Z]", section))
            target_size = self.MAX_CHUNK_SIZE if not has_break else self.MIN_CHUNK_SIZE

            # If section would exceed target size
            if current_length + len(section) > target_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                # Split large sections by sentences
                if len(section) > target_size:
                    sentences = re.split(r"(?<=[.!?])\s+", section)
                    temp_chunk = []
                    temp_length = 0

                    for sentence in sentences:
                        if temp_length + len(sentence) > target_size:
                            if temp_chunk:
                                chunks.append(" ".join(temp_chunk))
                                # Keep last sentence for minimal overlap
                                temp_chunk = (
                                    [temp_chunk[-1]] if len(temp_chunk) > 1 else []
                                )
                                temp_length = sum(len(s) + 1 for s in temp_chunk)
                            temp_chunk.append(sentence)
                            temp_length += len(sentence) + 1
                        else:
                            temp_chunk.append(sentence)
                            temp_length += len(sentence) + 1

                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                else:
                    chunks.append(section)

                current_chunk = []
                current_length = 0
            else:
                current_chunk.append(section)
                current_length += len(section)

        # Add remaining content
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)

        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks

    def process_document(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[ProcessedChunk]:
        """
        Process a document into chunks with embeddings and metadata.
        Uses adaptive chunk sizing based on content structure.
        """
        try:
            doc_id = f"doc_{int(time.time())}_{metadata.get('source', 'unknown')}"
            logger.info(f"Starting to process document with ID: {doc_id}")
            logger.info(f"Document content length: {len(content)} characters")

            # Split document into chunks
            chunks = self._split_into_semantic_chunks(content)
            logger.info(f"Split document into {len(chunks)} chunks")

            # Process each chunk
            processed_chunks = []
            for idx, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self._get_embedding(chunk)

                    # Create chunk metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "doc_id": doc_id,
                            "chunk_length": len(chunk),
                        }
                    )

                    processed_chunks.append(
                        ProcessedChunk(
                            content=chunk,
                            embedding=embedding,
                            metadata=chunk_metadata,
                            similarity_score=0.0,
                        )
                    )
                    logger.info(f"Successfully processed chunk {idx}")

                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {idx}: {str(chunk_error)}")
                    continue

            logger.info(f"Completed processing {len(processed_chunks)} chunks")
            return processed_chunks

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            raise
