from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from .job_parse import ParseDefaults

class HealthResponse(BaseModel):
    status: str
    device: str
    fp16: bool
    embed_model: str
    rerank_model: str

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    normalize: bool = True

class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    dim: int

class RerankRequest(BaseModel):
    query: str
    candidates: List[str] = Field(..., min_items=1)
    normalize: bool = True

class RerankResponse(BaseModel):
    scores: List[float]  # aligned with candidates order

class PairRerankRequest(BaseModel):
    pairs: List[List[str]] = Field(..., min_items=1, description="Each pair = [query, candidate]")
    normalize: bool = True

class PairRerankResponse(BaseModel):
    scores: List[float]
class ParseJobRequest(BaseModel):
    rawText: str = Field(min_length=10)
    defaults: ParseDefaults = Field(default_factory=ParseDefaults)


class ParseJobResponse(BaseModel):
    detectedSource: str
    suggested: Dict[str, Any]
    meta: Dict[str, Any]  # confidence, warnings