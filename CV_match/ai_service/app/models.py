import os
from typing import Optional
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from .config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, DEVICE, USE_FP16

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_embedder: Optional[BGEM3FlagModel] = None
_reranker: Optional[FlagReranker] = None

def get_embedder() -> BGEM3FlagModel:
    global _embedder
    if _embedder is None:
        _embedder = BGEM3FlagModel(
            EMBED_MODEL_NAME,
            use_fp16=USE_FP16,
            device=DEVICE,
        )
    return _embedder

def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(
            RERANK_MODEL_NAME,
            use_fp16=USE_FP16,
            device=DEVICE,
        )
    return _reranker
