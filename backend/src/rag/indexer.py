from typing import List
from pathlib import Path
import os

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, DocChunk

from src.rag_interface.base_datastore import DataItem
from src.rag_interface.base_indexer import BaseIndexer

ALLOWED_EXTS = {".pdf"}

class Indexer(BaseIndexer):
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()

        # Disable tokenizer parallelism (Docling can use HF tokenizers internally)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def index(self, document_paths: List[str]) -> List[DataItem]:
        items = []
        
        for document_path in document_paths:
        
            p = Path(document_path)

            # Skip hidden files like .DS_Store and only accept PDFs
            if p.name.startswith(".") or p.suffix.lower() not in ALLOWED_EXTS:
                continue

            document = self.converter.convert(str(p)).document
            chunks: List[DocChunk] = self.chunker.chunk(document)
            items.extend(self._items_from_chunks(chunks))
        return items
    
    def _items_from_chunks(self, chunks: List[DocChunk]) -> List[DataItem]:
        items = []
        for i, chunk in enumerate(chunks):
            content_headings = "## " + ", ".join(chunk.meta.headings)
            content_text = f"{content_headings}\n{chunk.text}"
            source = f"{chunk.meta.origin.filename}:{i}"
            item = DataItem(content=content_text, source=source)
            items.append(item)

        return items