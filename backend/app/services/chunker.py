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


class DocumentProcessor:
    """
    Handles document processing with an intelligent chunking strategy.
    This processor focuses on maintaining document structure and context
    while creating appropriately sized chunks.
    """

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

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections by natural breaks."""
        # No longer split by markdown headers
        sections = text.split("\n\n")  # Split by paragraph breaks
        logger.debug(f"Split document into {len(sections)} sections")
        return [s for s in sections if s.strip()]  # Remove empty sections

    # def _split_into_semantic_chunks(self, text: str) -> List[str]:
    #     """
    #     Split text into semantically meaningful chunks while respecting size limits.
    #     This method tries to keep related content together while ensuring chunks
    #     aren't too large.
    #     """
    #     # First split by major sections (headers)
    #     sections = self._split_into_sections(text)

    #     chunks = []
    #     for section_idx, section in enumerate(sections):
    #         logger.debug(f"Processing section {section_idx}")

    #         current_chunk = []
    #         current_length = 0

    #         # Split section into paragraphs
    #         paragraphs = section.split("\n\n")

    #         for para_idx, paragraph in enumerate(paragraphs):
    #             paragraph = paragraph.strip()
    #             if not paragraph:
    #                 continue

    #             logger.debug(
    #                 f"Processing paragraph {para_idx} in section {section_idx}"
    #             )
    #             logger.debug(f"Paragraph length: {len(paragraph)}")

    #             # If adding this paragraph would exceed max size
    #             if current_length + len(paragraph) > self.max_chunk_size:
    #                 # Save current chunk if it exists
    #                 if current_chunk:
    #                     chunk_text = "\n\n".join(current_chunk)
    #                     chunks.append(chunk_text)
    #                     # Create overlap by keeping some of the previous content
    #                     overlap_text = chunk_text[-self.chunk_overlap :]
    #                     current_chunk = [overlap_text]
    #                     current_length = len(overlap_text)

    #                 # Handle paragraphs that are themselves too long
    #                 if len(paragraph) > self.max_chunk_size:
    #                     logger.debug("Processing oversized paragraph")
    #                     # Split by sentences
    #                     sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    #                     temp_chunk = []
    #                     temp_length = 0

    #                     for sentence in enumerate(sentences):
    #                         if temp_length + len(sentence) > self.max_chunk_size:
    #                             if temp_chunk:
    #                                 chunk_text = " ".join(temp_chunk)
    #                                 chunks.append(chunk_text)

    #                                 # Create sentence-level overlap
    #                                 last_sentences = " ".join(
    #                                     temp_chunk[-2:]
    #                                 )  # Keep last 2 sentences
    #                                 temp_chunk = [last_sentences]
    #                                 temp_length = len(last_sentences)

    #                             temp_chunk = [sentence]
    #                             temp_length = len(sentence)
    #                         else:
    #                             temp_chunk.append(sentence)
    #                             temp_length += len(sentence) + 1  # +1 for space

    #                     if temp_chunk:
    #                         chunk_text = " ".join(temp_chunk)
    #                         chunks.append(chunk_text)

    #                 else:
    #                     current_chunk = [paragraph]
    #                     current_length = len(paragraph)
    #             else:
    #                 current_chunk.append(paragraph)
    #                 current_length += len(paragraph) + 2  # +2 for paragraph separator

    #         # Add any remaining content in the current chunk
    #         if current_chunk:
    #             chunk_text = "\n\n".join(current_chunk)
    #             chunks.append(chunk_text)

    #     logger.info(f"Created total of {len(chunks)} chunks")
    #     return chunks

    # app/services/chunker.py

    def _split_into_semantic_chunks(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks with consistent overlap.
        """
        sections = self._split_into_sections(text)
        chunks = []

        for section_idx, section in enumerate(sections):
            logger.debug(f"Processing section {section_idx + 1} of {len(sections)}")

            paragraphs = section.split("\n\n")
            current_chunk = []
            current_length = 0

            for para_idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                paragraph_length = len(paragraph)

                # If this paragraph would exceed the chunk size
                if current_length + paragraph_length > self.max_chunk_size:
                    if current_chunk:
                        # Save current chunk
                        chunk_text = "\n\n".join(current_chunk)
                        chunks.append(chunk_text)
                        logger.debug(f"Created chunk of length {len(chunk_text)}")

                        # Keep overlap from the end of previous chunk
                        overlap_text = chunk_text[-self.chunk_overlap :]
                        current_chunk = [overlap_text]
                        current_length = len(overlap_text)

                    # Handle large paragraphs
                    if paragraph_length > self.max_chunk_size:
                        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                        sentence_chunk = []
                        sentence_length = 0

                        for sentence in sentences:
                            if sentence_length + len(sentence) > self.max_chunk_size:
                                if sentence_chunk:
                                    chunks.append(" ".join(sentence_chunk))

                                    # Keep last sentence for overlap
                                    sentence_chunk = [sentence_chunk[-1], sentence]
                                    sentence_length = sum(
                                        len(s) for s in sentence_chunk
                                    )
                            else:
                                sentence_chunk.append(sentence)
                                sentence_length += len(sentence) + 1

                        if sentence_chunk:
                            chunks.append(" ".join(sentence_chunk))
                    else:
                        current_chunk = [paragraph]
                        current_length = paragraph_length
                else:
                    current_chunk.append(paragraph)
                    current_length += paragraph_length + 2  # +2 for paragraph separator

            # Add remaining content
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)

        logger.info(f"Split document into {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks):
            logger.debug(f"Chunk {idx + 1}: {len(chunk)} characters")

        return chunks

    def process_document(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[ProcessedChunk]:
        """
        Process a document into chunks with embeddings and metadata.
        This is the main method that coordinates the entire document processing pipeline.
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
                    logger.debug(f"Processing chunk {idx}")
                    logger.debug(f"Chunk {idx} length: {len(chunk)}")
                    logger.debug(f"Chunk {idx} preview: {chunk[:100]}...")

                    # Generate embedding
                    embedding = self._get_embedding(chunk)

                    # Create chunk metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "parent_id": doc_id,
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
