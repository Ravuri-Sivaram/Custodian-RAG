"""embedder: RAG embedding + retrieval components.

Qwen3-VL-Embedding-8B (dense, multimodal) + BM25 (sparse, regex-tokenized) + Qdrant (hybrid + ACL hard
filtering). Consumes chunker's Chunk[]; chunker's image_path/acl/source_indices/doc_type
fields are put to use here."""
from .acl import acl_admits, acl_split
from .config import EmbedConfig
from .embed import Embedder
from .rerank import Reranker
from .retriever import Retriever
from .types import Hit, User

__all__ = ["EmbedConfig", "Hit", "User", "Embedder", "Retriever", "Reranker", "acl_admits", "acl_split"]
