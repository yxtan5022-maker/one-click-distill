"""Data layer: ingestion, chunking, dataset writing."""

from .chunker import chunk_all, chunk_text
from .loader import DataLoadError, load_text_files

__all__ = ["chunk_all", "chunk_text", "load_text_files", "DataLoadError"]
