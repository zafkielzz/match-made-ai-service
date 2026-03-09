"""
Service layer cho business logic cua semantic matching.
"""

from typing import Dict, List, Tuple

import numpy as np

from .models import rerank
from .repository import DatabaseRepository
from .utils import (
    calculate_metrics,
    cosine_similarity_matrix,
    get_match_explanation,
    get_match_level,
    get_top_k_matches,
)


class SemanticMatchingService:
    def __init__(self):
        self.repository = DatabaseRepository()

    def _validate_chunks(
        self,
        responsibility_chunks: List[Dict],
        experience_chunks: List[Dict],
        job_id: str,
        cv_id: str,
    ) -> None:
        if not responsibility_chunks:
            raise ValueError(
                f"Khong tim thay responsibility chunks co embedding cho job_id={job_id}. "
                "Vui long chay embedding pipeline truoc."
            )

        if not experience_chunks:
            raise ValueError(
                f"Khong tim thay experience chunks co embedding cho cv_id={cv_id}. "
                "Vui long chay embedding pipeline truoc."
            )

    def _extract_vectors(self, chunks: List[Dict]) -> np.ndarray:
        vectors = [chunk["vec"] for chunk in chunks]
        return np.array(vectors, dtype=np.float32)

    def _build_responsibility_matches(
        self,
        responsibility_chunks: List[Dict],
        experience_chunks: List[Dict],
        similarity_matrix: np.ndarray,
    ) -> Tuple[List[Dict], List[float]]:
        matches = []
        best_scores = []

        for i, resp_chunk in enumerate(responsibility_chunks):
            similarity_row = similarity_matrix[i]
            best_idx = int(np.argmax(similarity_row))
            best_score = float(similarity_row[best_idx])
            best_scores.append(best_score)

            best_match_chunk = experience_chunks[best_idx]
            match_level = get_match_level(best_score)
            top_3_matches = get_top_k_matches(similarity_row, experience_chunks, k=3)

            matches.append(
                {
                    "responsibility_chunk_id": resp_chunk["id"],
                    "responsibility_content": resp_chunk["content"],
                    "best_match": {
                        "experience_chunk_id": best_match_chunk["id"],
                        "experience_content": best_match_chunk["content"],
                        "score": best_score,
                        "score_percent": round(best_score * 100, 2),
                        "match_level": match_level,
                        "explanation": get_match_explanation(best_score, match_level),
                    },
                    "top_3_matches": top_3_matches,
                }
            )

        return matches, best_scores

    def _categorize_responsibilities(self, matches: List[Dict]) -> Dict[str, List[Dict]]:
        well_matched = []
        partially_matched = []
        unmatched = []

        for match in matches:
            score = match["best_match"]["score"]
            best_match = match["best_match"]
            summary_item = {
                "responsibility_chunk_id": match["responsibility_chunk_id"],
                "responsibility_content": match["responsibility_content"],
                "best_experience_chunk_id": best_match["experience_chunk_id"],
                "best_experience_content": best_match["experience_content"],
                "best_score": score,
                "match_level": match["best_match"]["match_level"],
            }

            if score >= 0.75:
                well_matched.append(summary_item)
            elif score >= 0.60:
                partially_matched.append(summary_item)
            else:
                unmatched.append(summary_item)

        return {
            "well_matched_responsibilities": well_matched,
            "partially_matched_responsibilities": partially_matched,
            "unmatched_responsibilities": unmatched,
        }

    def match_job_cv(self, job_id: str, cv_id: str) -> Dict:
        if not self.repository.check_job_exists(job_id):
            raise ValueError(f"Job voi id={job_id} khong ton tai")
        if not self.repository.check_cv_exists(cv_id):
            raise ValueError(f"CV voi id={cv_id} khong ton tai")

        responsibility_chunks = self.repository.get_responsibility_chunks(job_id)
        experience_chunks = self.repository.get_experience_chunks(cv_id)
        self._validate_chunks(responsibility_chunks, experience_chunks, job_id, cv_id)

        resp_vectors = self._extract_vectors(responsibility_chunks)
        exp_vectors = self._extract_vectors(experience_chunks)
        similarity_matrix = cosine_similarity_matrix(resp_vectors, exp_vectors)
        matches, best_scores = self._build_responsibility_matches(
            responsibility_chunks,
            experience_chunks,
            similarity_matrix,
        )
        metrics = calculate_metrics(best_scores)
        summary = self._categorize_responsibilities(matches)

        return {
            "job_id": job_id,
            "cv_id": cv_id,
            "metrics": {
                "overall_score": metrics["overall_score"],
                "overall_score_percent": round(metrics["overall_score"] * 100, 2),
                "coverage_score": metrics["coverage_score"],
                "coverage_score_percent": round(metrics["coverage_score"] * 100, 2),
                "strong_match_ratio": metrics["strong_match_ratio"],
                "strong_match_ratio_percent": round(metrics["strong_match_ratio"] * 100, 2),
                "final_score": metrics["final_score"],
                "final_score_percent": round(metrics["final_score"] * 100, 2),
            },
            "summary": {
                "total_responsibilities": len(responsibility_chunks),
                "total_experiences": len(experience_chunks),
                "well_matched_count": len(summary["well_matched_responsibilities"]),
                "partially_matched_count": len(summary["partially_matched_responsibilities"]),
                "unmatched_count": len(summary["unmatched_responsibilities"]),
                **summary,
            },
            "detailed_matches": matches,
        }


class HybridRetrievalService:
    def __init__(self):
        self.repository = DatabaseRepository()

    def _normalize_limits(self, retrieve_top_n: int, rerank_top_k: int) -> Tuple[int, int]:
        retrieve_top_n = max(1, int(retrieve_top_n))
        rerank_top_k = max(1, int(rerank_top_k))
        rerank_top_k = min(rerank_top_k, retrieve_top_n)
        return retrieve_top_n, rerank_top_k

    def _validate_source_vectors(self, source_vectors: Dict, source_id: str, source_label: str) -> None:
        if not source_vectors:
            raise ValueError(
                f"{source_label} voi id={source_id} khong ton tai hoac chua co vectors. "
                "Vui long chay embedding pipeline truoc."
            )
        if not source_vectors.get("problem_sparse_vector") or not source_vectors.get("capability_sparse_vector"):
            raise ValueError(
                f"{source_label} voi id={source_id} chua co sparse vectors. "
                "Vui long chay embedding pipeline voi hybrid mode."
            )
        if not source_vectors.get("problem_text"):
            raise ValueError(f"{source_label} voi id={source_id} chua co problem_text de rerank.")

    def _rerank_results(
        self,
        source_problem_text: str,
        rows: List[Dict],
        id_key: str,
        output_key: str,
        rerank_top_k: int,
    ) -> Dict:
        rerank_top_k = min(rerank_top_k, len(rows))
        rerank_rows = rows[:rerank_top_k]
        if not rerank_rows:
            return {"total_reranked": 0, "items": []}

        professional_scores = rerank(
            source_problem_text,
            [row.get("problem_text") or "" for row in rerank_rows],
            normalize=True,
        )

        items = []
        for row, professional_match_score in zip(rerank_rows, professional_scores):
            professional_match_score = float(professional_match_score)
            items.append(
                {
                    output_key: str(row[id_key]),
                    "score": professional_match_score,
                    "details": {
                        "professional_match_score": professional_match_score,
                        "retrieval_problem_score": float(row["problem_score"]),
                        "retrieval_capability_score": float(row["capability_score"]),
                        "retrieval_final_score": float(row["final_score"]),
                    },
                }
            )

        items.sort(key=lambda item: item["score"], reverse=True)
        return {"total_reranked": rerank_top_k, "items": items}

    def find_top_candidates(self, job_id: str, retrieve_top_n: int = 50, rerank_top_k: int = 20) -> Dict:
        retrieve_top_n, rerank_top_k = self._normalize_limits(retrieve_top_n, rerank_top_k)
        job_vectors = self.repository.get_job_vectors(job_id)
        self._validate_source_vectors(job_vectors, job_id, "Job")

        results = self.repository.find_top_candidates_by_vectors(
            problem_dense=job_vectors["problem_dense_vector"],
            problem_sparse=job_vectors["problem_sparse_vector"],
            capability_dense=job_vectors["capability_dense_vector"],
            capability_sparse=job_vectors["capability_sparse_vector"],
            top_k=retrieve_top_n,
        )

        reranked = self._rerank_results(
            source_problem_text=job_vectors["problem_text"],
            rows=results,
            id_key="cv_id",
            output_key="cv_id",
            rerank_top_k=rerank_top_k,
        )

        return {
            "job_id": job_id,
            "total_retrieved": len(results),
            "total_reranked": reranked["total_reranked"],
            "candidates": reranked["items"],
        }

    def find_top_jobs(self, cv_id: str, retrieve_top_n: int = 50, rerank_top_k: int = 20) -> Dict:
        retrieve_top_n, rerank_top_k = self._normalize_limits(retrieve_top_n, rerank_top_k)
        cv_vectors = self.repository.get_cv_vectors(cv_id)
        self._validate_source_vectors(cv_vectors, cv_id, "CV")

        results = self.repository.find_top_jobs_by_vectors(
            problem_dense=cv_vectors["problem_dense_vector"],
            problem_sparse=cv_vectors["problem_sparse_vector"],
            capability_dense=cv_vectors["capability_dense_vector"],
            capability_sparse=cv_vectors["capability_sparse_vector"],
            top_k=retrieve_top_n,
        )

        reranked = self._rerank_results(
            source_problem_text=cv_vectors["problem_text"],
            rows=results,
            id_key="job_id",
            output_key="job_id",
            rerank_top_k=rerank_top_k,
        )

        return {
            "cv_id": cv_id,
            "total_retrieved": len(results),
            "total_reranked": reranked["total_reranked"],
            "jobs": reranked["items"],
        }
