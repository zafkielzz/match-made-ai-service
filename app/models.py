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


def format_sparse_vector(lexical_weights: dict, threshold: float = 0.01) -> dict:
    """
    Chuyển đổi lexical weights từ model.encode() thành định dạng sparse vector.
    
    Args:
        lexical_weights: Dictionary {token_id: weight} từ model
        threshold: Ngưỡng để loại bỏ các token có weight thấp (default: 0.01)
    
    Returns:
        Dictionary với format: {"indices": [int, ...], "values": [float, ...]}
        Indices được sắp xếp theo thứ tự tăng dần
    """
    if not lexical_weights:
        return {"indices": [], "values": []}
    
    # Filter tokens có weight > threshold
    filtered_items = [(token_id, weight) for token_id, weight in lexical_weights.items() 
                      if weight > threshold]
    
    if not filtered_items:
        return {"indices": [], "values": []}
    
    # Sort theo token_id (indices) tăng dần
    filtered_items.sort(key=lambda x: x[0])
    
    indices = [int(token_id) for token_id, _ in filtered_items]
    values = [float(weight) for _, weight in filtered_items]
    
    return {"indices": indices, "values": values}


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
    _ = emb.encode(["warmup"], batch_size=1, return_dense=True, return_sparse=False)
    _ = rr.compute_score([("warmup query", "warmup doc")], normalize=True)


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    Returns dense vectors for a batch of texts.
    Uses batching for speed.
    
    NOTE: This function is kept for backward compatibility with embedding_pipeline.py
    For new code, use embed_texts_hybrid() to get both dense and sparse vectors.
    """
    emb = get_embedder()
    cleaned = [cap_text(t) for t in texts]

    # BGEM3FlagModel.encode usually returns a dict containing "dense_vecs"
    out = emb.encode(
        cleaned,
        batch_size=EMBED_BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
    )

    # Be defensive across versions
    if isinstance(out, dict):
        vecs = out.get("dense_vecs")
        if vecs is None:
            vecs = out.get("dense_embeddings")
        if vecs is None:
            vecs = out.get("embeddings")
        if vecs is None:
            raise ValueError(f"Unexpected embed output keys: {list(out.keys())}")
        return vecs.tolist() if hasattr(vecs, "tolist") else list(vecs)

    # Some versions may directly return vectors
    return out.tolist() if hasattr(out, "tolist") else list(out)


def embed_texts_hybrid(texts: Sequence[str]) -> List[dict]:
    """
    Returns both dense and sparse vectors for a batch of texts.
    Supports Hybrid Search (Dense + Sparse).
    
    Args:
        texts: List of input texts
    
    Returns:
        List of dicts, each containing:
        {
            "dense_vector": [float, ...],  # 1024-dim dense embedding
            "sparse_vector": {             # Sparse lexical weights
                "indices": [int, ...],
                "values": [float, ...]
            }
        }
    """
    emb = get_embedder()
    cleaned = [cap_text(t) for t in texts]

    # Encode with both dense and sparse
    out = emb.encode(
        cleaned,
        batch_size=EMBED_BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
    )

    if not isinstance(out, dict):
        raise ValueError(f"Expected dict output from model.encode, got {type(out)}")

    # Extract dense vectors
    dense_vecs = out.get("dense_vecs")
    if dense_vecs is None:
        dense_vecs = out.get("dense_embeddings")
    if dense_vecs is None:
        dense_vecs = out.get("embeddings")
    if dense_vecs is None:
        raise ValueError(f"Cannot find dense vectors in output keys: {list(out.keys())}")

    # Extract sparse vectors (lexical weights)
    lexical_weights = out.get("lexical_weights")
    if lexical_weights is None:
        lexical_weights = out.get("sparse_vecs")
    if lexical_weights is None:
        raise ValueError(f"Cannot find sparse vectors in output keys: {list(out.keys())}")

    # Convert to list if needed
    dense_list = dense_vecs.tolist() if hasattr(dense_vecs, "tolist") else list(dense_vecs)

    # Build result
    results = []
    for i, dense_vec in enumerate(dense_list):
        # lexical_weights is typically a dict per text or a list of dicts
        if isinstance(lexical_weights, list):
            lex_weights = lexical_weights[i] if i < len(lexical_weights) else {}
        else:
            # If it's a single dict (batch size 1), use it directly
            lex_weights = lexical_weights if len(cleaned) == 1 else {}

        sparse_vec = format_sparse_vector(lex_weights)
        
        results.append({
            "dense_vector": dense_vec,
            "sparse_vector": sparse_vec
        })

    return results


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
