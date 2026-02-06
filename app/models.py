import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from .config import (
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    DEVICE,
    USE_FP16,
    MAX_LENGTH,
    EMBED_BATCH_SIZE,
    RERANK_BATCH_SIZE,
)

# Avoid tokenizer spawning too many threads (helps stability in web servers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_embedder: Optional[BGEM3FlagModel] = None
_reranker: Optional[FlagReranker] = None


def _is_cuda(device: str) -> bool:
    d = (device or "").lower().strip()
    return d.startswith("cuda")


def _effective_fp16() -> bool:
    """
    FP16 only makes sense on CUDA. If config requests FP16 on CPU, ignore it.
    """
    return bool(USE_FP16 and _is_cuda(DEVICE) and torch.cuda.is_available())


def cap_text(text: Optional[str], max_chars: int = MAX_LENGTH) -> str:
    """
    Cap text by characters to keep rerank latency stable.
    MAX_LENGTH in your config is treated as max chars (not tokens).
    """
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars]


def get_embedder() -> BGEM3FlagModel:
    global _embedder
    if _embedder is None:
        fp16 = _effective_fp16()
        _embedder = BGEM3FlagModel(
            EMBED_MODEL_NAME,
            use_fp16=fp16,
            device=DEVICE,
        )
    return _embedder


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        fp16 = _effective_fp16()
        _reranker = FlagReranker(
            RERANK_MODEL_NAME,
            use_fp16=fp16,
            device=DEVICE,
        )
    return _reranker


def warmup_models() -> None:
    """
    Optional: call this once on app startup to avoid first-request cold start.
    """
    emb = get_embedder()
    rr = get_reranker()

    # Small warmup inputs
    _ = emb.encode(["warmup"], batch_size=1)
    _ = rr.compute_score([("warmup query", "warmup doc")], normalize=True)


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    Returns dense vectors for a batch of texts.
    Uses batching for speed.
    """
    emb = get_embedder()
    cleaned = [cap_text(t) for t in texts]

    # BGEM3FlagModel.encode usually returns a dict containing "dense_vecs"
    out = emb.encode(
        cleaned,
        batch_size=EMBED_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    # Be defensive across versions
    if isinstance(out, dict):
        vecs = out.get("dense_vecs") or out.get("dense_embeddings") or out.get("embeddings")
        if vecs is None:
            raise ValueError(f"Unexpected embed output keys: {list(out.keys())}")
        return vecs.tolist() if hasattr(vecs, "tolist") else list(vecs)

    # Some versions may directly return vectors
    return out.tolist() if hasattr(out, "tolist") else list(out)


def rerank(
    query: str,
    docs: Sequence[str],
    *,
    normalize: bool = True,
) -> List[float]:
    """
    Rerank docs against a query. Returns one score per doc.

    IMPORTANT:
    - Uses batching (RERANK_BATCH_SIZE)
    - Caps texts to stabilize latency
    """
    rr = get_reranker()
    q = cap_text(query)

    # Create pairs
    pairs: List[Tuple[str, str]] = [(q, cap_text(d)) for d in docs]

    # If docs is small, one shot is fine.
    if len(pairs) <= RERANK_BATCH_SIZE:
        scores = rr.compute_score(pairs, normalize=normalize)
        return list(scores)

    # Otherwise, batch manually to avoid OOM / latency spikes
    scores_all: List[float] = []
    for i in range(0, len(pairs), RERANK_BATCH_SIZE):
        batch = pairs[i : i + RERANK_BATCH_SIZE]
        scores = rr.compute_score(batch, normalize=normalize)
        scores_all.extend(list(scores))
    return scores_all


def log_runtime_info() -> None:
    """
    Helpful debug logs (call once on startup).
    """
    print("EMBED_MODEL_NAME:", EMBED_MODEL_NAME)
    print("RERANK_MODEL_NAME:", RERANK_MODEL_NAME)
    print("DEVICE:", DEVICE)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("USE_FP16 (config):", USE_FP16)
    print("USE_FP16 (effective):", _effective_fp16())
    print("MAX_LENGTH (chars cap):", MAX_LENGTH)
    print("EMBED_BATCH_SIZE:", EMBED_BATCH_SIZE)
    print("RERANK_BATCH_SIZE:", RERANK_BATCH_SIZE)
