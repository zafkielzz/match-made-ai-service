from fastapi import FastAPI, HTTPException
import numpy as np
import faiss  # only for normalize_L2; no indexing here
from .schemas import (
    HealthResponse, EmbedRequest, EmbedResponse,
    RerankRequest, RerankResponse,
    PairRerankRequest, PairRerankResponse
)
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from .config import DEVICE, USE_FP16, EMBED_MODEL_NAME, RERANK_MODEL_NAME, MAX_LENGTH, EMBED_BATCH_SIZE, RERANK_BATCH_SIZE
from .models import get_embedder, get_reranker, warmup_models, log_runtime_info, cap_text
load_dotenv() 
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log_runtime_info()
        warmup_models()
    except Exception as e:
        # Không kill server; log ra để debug
        print("Warmup failed:", repr(e))
    yield

app = FastAPI(lifespan=lifespan)
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
    """
    Rerank candidates based on a single query using BGE reranker.
    
    - **query**: The reference text (e.g., CV text or job description)
    - **candidates**: List of texts to rank against the query
    - **top_k**: Number of top candidates to rerank (default: 20)
    - **normalize**: Whether to normalize scores (default: true)
    
    Returns scores in the same order as input candidates (not sorted).
    """
    # Validation: query và candidates không được rỗng
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not req.candidates or len(req.candidates) == 0:
        raise HTTPException(status_code=400, detail="Candidates cannot be empty")
    
    reranker = get_reranker()
    try:
        # Cap text để ổn định latency
        query = cap_text(req.query, max_chars=MAX_LENGTH)
        
        # Limit candidates to top_k và cap text
        candidates = [cap_text(c, max_chars=MAX_LENGTH) for c in req.candidates[:req.top_k]]
        
        # Build pairs: [[query, candidate], ...]
        pairs = [[query, c] for c in candidates]
        
        # Batch rerank (compute_score tự động batch với batch_size)
        scores = reranker.compute_score(
            pairs,
            normalize=req.normalize,
            batch_size=RERANK_BATCH_SIZE
        )
        
        # Return scores in original order (not sorted)
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
