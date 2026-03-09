"""
Utility functions cho semantic similarity và scoring.
"""

import numpy as np
from typing import List, Tuple


def cosine_similarity_matrix(vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
    """
    Tính cosine similarity matrix giữa 2 tập vectors.
    
    Args:
        vectors_a: Array shape (n, dim) - responsibility vectors
        vectors_b: Array shape (m, dim) - experience vectors
    
    Returns:
        Similarity matrix shape (n, m) với giá trị từ -1 đến 1
    """
    # Normalize vectors
    norm_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)
    
    # Tránh chia cho 0
    norm_a = np.where(norm_a == 0, 1, norm_a)
    norm_b = np.where(norm_b == 0, 1, norm_b)
    
    vectors_a_normalized = vectors_a / norm_a
    vectors_b_normalized = vectors_b / norm_b
    
    # Cosine similarity = dot product của normalized vectors
    similarity = np.dot(vectors_a_normalized, vectors_b_normalized.T)
    
    return similarity


def get_match_level(score: float) -> str:
    """
    Xác định mức độ match dựa trên similarity score.
    
    Args:
        score: Cosine similarity score (0-1)
    
    Returns:
        Match level string
    """
    if score >= 0.85:
        return "very_strong_match"
    elif score >= 0.75:
        return "strong_match"
    elif score >= 0.60:
        return "moderate_match"
    elif score >= 0.45:
        return "weak_match"
    else:
        return "low_match"


def get_match_explanation(score: float, match_level: str) -> str:
    """
    Tạo explanation ngắn gọn cho mức độ match.
    
    Args:
        score: Similarity score
        match_level: Match level string
    
    Returns:
        Human-readable explanation
    """
    explanations = {
        "very_strong_match": "Kinh nghiệm rất phù hợp với yêu cầu công việc",
        "strong_match": "Kinh nghiệm phù hợp tốt với yêu cầu công việc",
        "moderate_match": "Kinh nghiệm có liên quan đến yêu cầu công việc",
        "weak_match": "Kinh nghiệm có một số điểm tương đồng với yêu cầu",
        "low_match": "Kinh nghiệm ít liên quan đến yêu cầu công việc"
    }
    return explanations.get(match_level, "Không xác định được mức độ phù hợp")


def calculate_metrics(best_scores: List[float]) -> dict:
    """
    Tính toán các metrics tổng quan.
    
    Args:
        best_scores: List các best similarity scores cho mỗi responsibility
    
    Returns:
        Dictionary chứa các metrics
    """
    if not best_scores:
        return {
            "overall_score": 0.0,
            "coverage_score": 0.0,
            "strong_match_ratio": 0.0,
            "final_score": 0.0
        }
    
    best_scores_array = np.array(best_scores)
    total = len(best_scores)
    
    # Overall score = mean của best scores
    overall_score = float(np.mean(best_scores_array))
    
    # Coverage score = tỷ lệ responsibilities có score >= 0.60
    coverage_count = np.sum(best_scores_array >= 0.60)
    coverage_score = float(coverage_count / total)
    
    # Strong match ratio = tỷ lệ responsibilities có score >= 0.75
    strong_count = np.sum(best_scores_array >= 0.75)
    strong_match_ratio = float(strong_count / total)
    
    # Final score = weighted combination
    final_score = 0.7 * overall_score + 0.3 * coverage_score
    
    return {
        "overall_score": overall_score,
        "coverage_score": coverage_score,
        "strong_match_ratio": strong_match_ratio,
        "final_score": final_score
    }


def get_top_k_matches(
    similarity_row: np.ndarray,
    experience_chunks: List[dict],
    k: int = 3
) -> List[dict]:
    """
    Lấy top K experience chunks có similarity cao nhất.
    
    Args:
        similarity_row: Array similarity scores cho 1 responsibility
        experience_chunks: List các experience chunk dicts
        k: Số lượng top matches cần lấy
    
    Returns:
        List top K matches với score và content
    """
    # Lấy indices của top k scores
    top_k_indices = np.argsort(similarity_row)[-k:][::-1]
    
    matches = []
    for idx in top_k_indices:
        score = float(similarity_row[idx])
        chunk = experience_chunks[idx]
        matches.append({
            "chunk_id": chunk["id"],
            "content": chunk["content"],
            "score": score,
            "score_percent": round(score * 100, 2),
            "match_level": get_match_level(score)
        })
    
    return matches
