from fastapi import FastAPI, HTTPException
import numpy as np
import faiss  # only for normalize_L2; no indexing here

from .schemas import (
    HealthResponse, EmbedRequest, EmbedResponse,
    RerankRequest, RerankResponse,
    PairRerankRequest, PairRerankResponse
)
from .config import DEVICE, USE_FP16, EMBED_MODEL_NAME, RERANK_MODEL_NAME, MAX_LENGTH, EMBED_BATCH_SIZE, RERANK_BATCH_SIZE
from .models import get_embedder, get_reranker

app = FastAPI(title="AI Inference Service (Embedding + Rerank)")

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=DEVICE,
        fp16=USE_FP16,
        embed_model=EMBED_MODEL_NAME,
        rerank_model=RERANK_MODEL_NAME
    )

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    embedder = get_embedder()
    try:
        out = embedder.encode(
            req.texts,
            batch_size=EMBED_BATCH_SIZE,
            max_length=MAX_LENGTH
        )
        vecs = np.asarray(out["dense_vecs"], dtype="float32")
        if req.normalize:
            faiss.normalize_L2(vecs)
        return EmbedResponse(vectors=vecs.tolist(), dim=int(vecs.shape[1]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    reranker = get_reranker()
    try:
        pairs = [[req.query, c] for c in req.candidates]
        scores = reranker.compute_score(
            pairs,
            normalize=req.normalize,
            batch_size=RERANK_BATCH_SIZE
        )
        return RerankResponse(scores=[float(s) for s in scores])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}")

@app.post("/rerank_pairs", response_model=PairRerankResponse)
def rerank_pairs(req: PairRerankRequest):
    
    
    reranker = get_reranker()
    try:
        # Validate pairs shape
        for p in req.pairs:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError("Each pair must be [query, candidate]")
        scores = reranker.compute_score(
            req.pairs,
            normalize=req.normalize,
            batch_size=RERANK_BATCH_SIZE
        )
        return PairRerankResponse(scores=[float(s) for s in scores])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request or rerank failed: {e}")
