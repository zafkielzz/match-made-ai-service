from contextlib import asynccontextmanager
from typing import Union

import faiss  # only for normalize_L2; no indexing here
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from .config import (
    DEVICE,
    EMBED_BATCH_SIZE,
    EMBED_MODEL_NAME,
    MAX_LENGTH,
    RERANK_BATCH_SIZE,
    RERANK_MODEL_NAME,
    USE_FP16,
)
from .models import cap_text, embed_texts_hybrid, get_embedder, get_reranker, log_runtime_info, warmup_models
from .schemas import (
    EmbedRequest,
    EmbedResponse,
    FindCandidatesForJobResponse,
    FindCandidatesRequest,
    FindCandidatesResponse,
    FindJobsForCVResponse,
    HealthResponse,
    HybridEmbedResponse,
    PairRerankRequest,
    PairRerankResponse,
    RerankRequest,
    RerankResponse,
    SemanticMatchRequest,
    SemanticMatchResponse,
)
from .service import HybridRetrievalService, SemanticMatchingService

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log_runtime_info()
        warmup_models()
    except Exception as e:
        print("Warmup failed:", repr(e))
    yield


app = FastAPI(lifespan=lifespan)
semantic_service = SemanticMatchingService()
retrieval_service = HybridRetrievalService()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=DEVICE,
        fp16=USE_FP16,
        embed_model=EMBED_MODEL_NAME,
        rerank_model=RERANK_MODEL_NAME,
    )


@app.post("/embed", response_model=Union[EmbedResponse, HybridEmbedResponse])
def embed(req: EmbedRequest):
    embedder = get_embedder()
    try:
        if req.return_sparse:
            results = embed_texts_hybrid(req.texts)
            if req.normalize:
                for result in results:
                    dense_vec = np.array(result["dense_vector"], dtype="float32").reshape(1, -1)
                    faiss.normalize_L2(dense_vec)
                    result["dense_vector"] = dense_vec[0].tolist()

            embeddings = [{"dense_vector": r["dense_vector"], "sparse_vector": r["sparse_vector"]} for r in results]
            dense_dim = len(results[0]["dense_vector"]) if results else 0
            return HybridEmbedResponse(embeddings=embeddings, dense_dim=dense_dim)

        out = embedder.encode(
            req.texts,
            batch_size=EMBED_BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
        )
        vecs = np.asarray(out["dense_vecs"], dtype="float32")
        if req.normalize:
            faiss.normalize_L2(vecs)
        return EmbedResponse(vectors=vecs.tolist(), dim=int(vecs.shape[1]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")




@app.post("/semantic_match", response_model=SemanticMatchResponse)
def semantic_match(req: SemanticMatchRequest):
    try:
        return semantic_service.match_job_cv(req.job_id, req.cv_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic matching failed: {str(e)}")


@app.post("/find-top-candidates", response_model=FindCandidatesResponse)
@app.post("/find-best-match", response_model=FindCandidatesResponse)
def find_top_candidates(req: FindCandidatesRequest):
    """
    Retrieve top N by multi-vector hybrid search, then rerank only on problem_text.
    professional_match_score = rerank(JD.problem_text, CV.problem_text).
    capability_text is used only in retrieval, not in reranking.
    """
    try:
        retrieve_top_n = req.retrieve_top_n or req.top_k or 50
        rerank_top_k = req.rerank_top_k or min(retrieve_top_n, req.top_k or 20)

        if req.type == "job":
            if not req.job_id:
                raise HTTPException(status_code=400, detail="job_id is required when type='job'")
            result = retrieval_service.find_top_candidates(req.job_id, retrieve_top_n, rerank_top_k)
            return FindCandidatesForJobResponse(**result)

        if req.type == "cv":
            if not req.cv_id:
                raise HTTPException(status_code=400, detail="cv_id is required when type='cv'")
            result = retrieval_service.find_top_jobs(req.cv_id, retrieve_top_n, rerank_top_k)
            payload = {
                "cv_id": result["cv_id"],
                "total_retrieved": result["total_retrieved"],
                "total_reranked": result["total_reranked"],
                "candidates": result["jobs"],
            }
            return FindJobsForCVResponse(**payload)

        raise HTTPException(status_code=400, detail="type must be either 'job' or 'cv'")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid retrieval failed: {str(e)}")
