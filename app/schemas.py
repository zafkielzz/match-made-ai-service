from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status (e.g., 'ok').")
    device: str = Field(..., description="Compute device in use (e.g., 'cpu' or 'cuda').")
    fp16: bool = Field(..., description="Whether FP16 (half precision) is enabled for model inference.")
    embed_model: str = Field(..., description="Identifier/name of the embedding model currently loaded.")
    rerank_model: str = Field(..., description="Identifier/name of the rerank (cross-encoder) model currently loaded.")


class EmbedRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        description="List of input texts to embed. Each item should be a single-line, normalized text string."
    )
    normalize: bool = Field(
        default=True,
        description=(
            "If true, L2-normalize each embedding vector (unit-length). "
            "This is commonly used so cosine similarity can be computed via dot product."
        )
    )


class EmbedResponse(BaseModel):
    vectors: List[List[float]] = Field(
        ...,
        description="Embedding vectors aligned with the input texts order."
    )
    dim: int = Field(
        ...,
        description="Embedding dimensionality (length of each vector)."
    )


class RerankRequest(BaseModel):
    query: str = Field(
        ...,
        description="Query text used to rerank candidates (e.g., JOB_TEXT or CV_TEXT depending on your direction)."
    )
    candidates: List[str] = Field(
        ...,
        min_items=1,
        description="List of candidate texts to score against the query. Order is preserved in the response."
    )
    top_k: int = Field(
        default=20,
        ge=1,
        description="Number of top candidates to rerank (applied after any pre-filtering on the server side)."
    )
    normalize: bool = Field(
        default=True,
        description=(
            "If true, normalize rerank scores within this request to the range [0, 1] for easier comparison. "
            "Note: normalized scores are relative to the current batch, not a calibrated probability."
        )
    )


class RerankResponse(BaseModel):
    scores: List[float] = Field(
        ...,
        description="Rerank scores aligned with the candidates order (same index as input candidates)."
    )


class PairRerankRequest(BaseModel):
    pairs: List[List[str]] = Field(
        ...,
        description=(
            "List of text pairs to rerank. Each item is a 2-element list: "
            "[query_text, candidate_text]. Use this when you already have explicit pairs to score."
        ),
        example=[
            [
                "AI Engineer Intern focusing on Computer Vision and Deep Learning",
                "Candidate with experience in CNN-based models, PyTorch, and image classification",
            ],
            [
                "NLP Engineer Intern working on text classification and summarization",
                "Candidate experienced in tokenization, transformers, and NLP evaluation",
            ],
        ],
    )
    normalize: bool = Field(
        default=True,
        description=(
            "If true, normalize rerank scores within this request to the range [0, 1]. "
            "This helps UI display and mixing with other heuristics, but remains batch-relative."
        ),
        example=True,
    )


class PairRerankResponse(BaseModel):
    scores: List[float] = Field(
        ...,
        description="Rerank scores aligned with the input pairs order (same index as input pairs)."
    )
