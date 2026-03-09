from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status (e.g., 'ok').")
    device: str = Field(..., description="Compute device in use (e.g., 'cpu' or 'cuda').")
    fp16: bool = Field(..., description="Whether FP16 (half precision) is enabled for model inference.")
    embed_model: str = Field(..., description="Identifier/name of the embedding model currently loaded.")
    rerank_model: str = Field(..., description="Identifier/name of the rerank (cross-encoder) model currently loaded.")


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of input texts to embed.")
    normalize: bool = Field(default=True, description="Whether to L2-normalize each embedding vector.")
    return_sparse: bool = Field(default=False, description="If true, return both dense and sparse vectors.")


class SparseVector(BaseModel):
    indices: List[int] = Field(..., description="Token indices in vocabulary (sorted ascending)")
    values: List[float] = Field(..., description="Weights for each token")


class HybridEmbedding(BaseModel):
    dense_vector: List[float] = Field(..., description="Dense embedding vector (1024-dim)")
    sparse_vector: SparseVector = Field(..., description="Sparse lexical weights")


class EmbedResponse(BaseModel):
    vectors: List[List[float]] = Field(..., description="Embedding vectors aligned with the input texts order.")
    dim: int = Field(..., description="Embedding dimensionality.")


class HybridEmbedResponse(BaseModel):
    embeddings: List[HybridEmbedding] = Field(..., description="Hybrid embeddings aligned with input order.")
    dense_dim: int = Field(..., description="Dense embedding dimensionality.")


class RerankRequest(BaseModel):
    query: str = Field(..., description="Query text used to rerank candidates.")
    candidates: List[str] = Field(..., min_items=1, description="Candidate texts to score against the query.")
    top_k: int = Field(default=20, ge=1, description="Number of top candidates to rerank.")
    normalize: bool = Field(default=True, description="Whether to normalize rerank scores to [0, 1].")


class RerankResponse(BaseModel):
    scores: List[float] = Field(..., description="Rerank scores aligned with the candidates order.")


class PairRerankRequest(BaseModel):
    pairs: List[List[str]] = Field(..., description="List of [query_text, candidate_text] pairs.")
    normalize: bool = Field(default=True, description="Whether to normalize rerank scores to [0, 1].")


class PairRerankResponse(BaseModel):
    scores: List[float] = Field(..., description="Rerank scores aligned with the input pairs order.")


class JobSkills(BaseModel):
    must_have: List[str] = Field(default_factory=list, description="Required skills for the job")
    nice_to_have: List[str] = Field(default_factory=list, description="Preferred but not required skills")
    soft_skills: List[str] = Field(default_factory=list, description="Soft skills like communication, leadership")


class CVData(BaseModel):
    extracted_skills: List[str] = Field(..., description="List of skills extracted from CV")


class JobData(BaseModel):
    extracted_skills: JobSkills = Field(..., description="Job skills categorized by level")


class ExplainRequest(BaseModel):
    cv: CVData = Field(..., description="CV data with extracted skills")
    job: JobData = Field(..., description="Job data with extracted skills")


class MatchedSkills(BaseModel):
    must_have: List[str] = Field(default_factory=list, description="Matched must-have skills")
    nice_to_have: List[str] = Field(default_factory=list, description="Matched nice-to-have skills")
    soft_skills: List[str] = Field(default_factory=list, description="Matched soft skills")


class MissingSkills(BaseModel):
    must_have: List[str] = Field(default_factory=list, description="Missing must-have skills")
    nice_to_have: List[str] = Field(default_factory=list, description="Missing nice-to-have skills")
    soft_skills: List[str] = Field(default_factory=list, description="Missing soft skills")


class ExplainStats(BaseModel):
    must_have_coverage: float = Field(..., description="Percentage of must-have skills matched")
    overall_coverage: float = Field(..., description="Overall skill coverage across all categories")
    critical_gap: bool = Field(..., description="True if any must-have skills are missing")


class ExplainResponse(BaseModel):
    matched: MatchedSkills = Field(..., description="Skills that match between CV and Job")
    missing: MissingSkills = Field(..., description="Skills required by Job but missing in CV")
    stats: ExplainStats = Field(..., description="Coverage statistics")


class SemanticMatchRequest(BaseModel):
    job_id: str = Field(..., description="ID cua job can so sanh (UUID)")
    cv_id: str = Field(..., description="ID cua CV can so sanh (UUID)")


class BestMatchDetail(BaseModel):
    experience_chunk_id: str = Field(..., description="ID cua experience chunk match tot nhat (UUID)")
    experience_content: str = Field(..., description="Noi dung cua experience chunk")
    score: float = Field(..., description="Cosine similarity score (0-1)")
    score_percent: float = Field(..., description="Score dang phan tram")
    match_level: str = Field(..., description="Muc do match")
    explanation: str = Field(..., description="Giai thich ngan ve muc do match")


class TopMatchDetail(BaseModel):
    chunk_id: str = Field(..., description="ID cua chunk (UUID)")
    content: str = Field(..., description="Noi dung chunk")
    score: float = Field(..., description="Similarity score")
    score_percent: float = Field(..., description="Score dang phan tram")
    match_level: str = Field(..., description="Muc do match")


class ResponsibilityMatch(BaseModel):
    responsibility_chunk_id: str = Field(..., description="ID cua responsibility chunk (UUID)")
    responsibility_content: str = Field(..., description="Noi dung responsibility")
    best_match: BestMatchDetail = Field(..., description="Experience chunk match tot nhat")
    top_3_matches: List[TopMatchDetail] = Field(..., description="Top 3 experience chunks gan nhat")


class ResponsibilitySummaryItem(BaseModel):
    responsibility_chunk_id: str
    responsibility_content: str
    best_experience_chunk_id: str = Field(..., description="ID cua experience chunk match tot nhat")
    best_experience_content: str = Field(..., description="Noi dung experience chunk match tot nhat")
    best_score: float
    match_level: str


class MatchingMetrics(BaseModel):
    overall_score: float = Field(..., description="Diem trung binh cua best matches (0-1)")
    overall_score_percent: float = Field(..., description="Overall score dang phan tram")
    coverage_score: float = Field(..., description="Ty le responsibilities duoc cover tot")
    coverage_score_percent: float = Field(..., description="Coverage score dang phan tram")
    strong_match_ratio: float = Field(..., description="Ty le responsibilities co strong match")
    strong_match_ratio_percent: float = Field(..., description="Strong match ratio dang phan tram")
    final_score: float = Field(..., description="Diem tong hop")
    final_score_percent: float = Field(..., description="Final score dang phan tram")


class MatchingSummary(BaseModel):
    total_responsibilities: int = Field(..., description="Tong so responsibility chunks")
    total_experiences: int = Field(..., description="Tong so experience chunks")
    well_matched_count: int = Field(..., description="So responsibilities match tot")
    partially_matched_count: int = Field(..., description="So responsibilities match vua phai")
    unmatched_count: int = Field(..., description="So responsibilities match yeu")
    well_matched_responsibilities: List[ResponsibilitySummaryItem] = Field(..., description="Danh sach responsibilities match tot")
    partially_matched_responsibilities: List[ResponsibilitySummaryItem] = Field(..., description="Danh sach responsibilities match vua")
    unmatched_responsibilities: List[ResponsibilitySummaryItem] = Field(..., description="Danh sach responsibilities match yeu")


class SemanticMatchResponse(BaseModel):
    job_id: str = Field(..., description="ID cua job")
    cv_id: str = Field(..., description="ID cua CV")
    metrics: MatchingMetrics = Field(..., description="Cac metrics tong quan")
    summary: MatchingSummary = Field(..., description="Tom tat ket qua matching")
    detailed_matches: List[ResponsibilityMatch] = Field(..., description="Chi tiet matching cho tung responsibility")


class FindCandidatesRequest(BaseModel):
    job_id: Optional[str] = Field(None, description="ID cua job (neu tim candidates)")
    cv_id: Optional[str] = Field(None, description="ID cua CV (neu tim jobs)")
    type: str = Field(..., description="Loai tim kiem: 'job' hoac 'cv'")
    retrieve_top_n: Optional[int] = Field(default=None, ge=1, le=200, description="So luong ket qua retrieve ban dau")
    rerank_top_k: Optional[int] = Field(default=None, ge=1, le=200, description="So luong ket qua top retrieve duoc dua vao rerank")
    top_k: Optional[int] = Field(default=None, ge=1, le=200, description="Backward compatible alias")


class MatchScoreDetail(BaseModel):
    professional_match_score: float = Field(..., description="Diem rerank cho cap problem_text JD vs CV (0-1)")
    retrieval_problem_score: Optional[float] = Field(None, description="Diem hybrid retrieval cho problem space")
    retrieval_capability_score: Optional[float] = Field(None, description="Diem hybrid retrieval cho capability space")
    retrieval_final_score: Optional[float] = Field(None, description="Diem tong hop o buoc retrieval truoc rerank")


class CVMatchResult(BaseModel):
    cv_id: str = Field(..., description="ID cua CV candidate")
    score: float = Field(..., description="Diem cuoi cung sau rerank")
    details: MatchScoreDetail = Field(..., description="Chi tiet diem so")


class JobMatchResult(BaseModel):
    job_id: str = Field(..., description="ID cua Job candidate")
    score: float = Field(..., description="Diem cuoi cung sau rerank")
    details: MatchScoreDetail = Field(..., description="Chi tiet diem so")


class FindCandidatesForJobResponse(BaseModel):
    job_id: str = Field(..., description="ID cua job nguon")
    total_retrieved: int = Field(..., description="So luong CVs tim duoc tu retrieval")
    total_reranked: int = Field(..., description="So luong CVs duoc dua vao rerank")
    candidates: List[CVMatchResult] = Field(..., description="Danh sach CVs duoc sap xep lai sau rerank")


class FindJobsForCVResponse(BaseModel):
    cv_id: str = Field(..., description="ID cua CV nguon")
    total_retrieved: int = Field(..., description="So luong Jobs tim duoc tu retrieval")
    total_reranked: int = Field(..., description="So luong Jobs duoc dua vao rerank")
    candidates: List[JobMatchResult] = Field(..., description="Danh sach Jobs duoc sap xep lai sau rerank")


FindCandidatesResponse = Union[FindCandidatesForJobResponse, FindJobsForCVResponse]
