"""
RAG модуль для работы с документами и векторным поиском.
"""

from rag.rag_pipeline import RAGPipeline
from rag.chunker_integration import ChunkerIntegration
from rag.giga_embeddings import GigaEmbedding
from rag.reranker import ChatCompletionsReranker
from rag.vector_store import QdrantVectorStoreManager
from rag.indexer import DocumentIndexer
from rag.retriever import DocumentRetriever

__all__ = [
    "RAGPipeline",
    "ChunkerIntegration",
    "GigaEmbedding",
    "ChatCompletionsReranker",
    "QdrantVectorStoreManager",
    "DocumentIndexer",
    "DocumentRetriever",
]

__version__ = "0.1.0"
