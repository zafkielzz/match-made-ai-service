"""
Repository layer de tuong tac voi database.
"""

import json
import os
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

SPARSE_DIMENSION = 30522
MIN_PROBLEM_SCORE = 0.2


def parse_pgvector(vec_str: str) -> List[float]:
    if isinstance(vec_str, list):
        return vec_str

    if isinstance(vec_str, str):
        vec_str = vec_str.strip()
        if vec_str.startswith('[') and vec_str.endswith(']'):
            vec_str = vec_str[1:-1]
        return [float(x.strip()) for x in vec_str.split(',') if x.strip()]

    return list(vec_str)


def parse_sparsevec(sparse_str) -> Dict:
    if isinstance(sparse_str, dict):
        return sparse_str

    if not sparse_str:
        return {"indices": [], "values": []}

    sparse_str = str(sparse_str).strip()
    if sparse_str in {"{}/30000", "{}", f"{{}}/{SPARSE_DIMENSION}"}:
        return {"indices": [], "values": []}

    try:
        if sparse_str.startswith('{"'):
            parsed = json.loads(sparse_str)
            if isinstance(parsed, dict):
                return {
                    "indices": [int(x) for x in parsed.get("indices", [])],
                    "values": [float(x) for x in parsed.get("values", [])],
                }

        vector_part = sparse_str.split('/', 1)[0].strip('{}')
        if not vector_part:
            return {"indices": [], "values": []}

        indices = []
        values = []
        for pair in vector_part.split(','):
            if ':' not in pair:
                continue
            idx, val = pair.split(':', 1)
            indices.append(int(idx.strip()))
            values.append(float(val.strip()))

        return {"indices": indices, "values": values}
    except Exception as e:
        print(f"Warning: Failed to parse sparse vector '{sparse_str}': {e}")
        return {"indices": [], "values": []}


def format_sparsevec_query(sparse_vector: Dict) -> str:
    if not sparse_vector:
        return f"{{}}/{SPARSE_DIMENSION}"

    indices = sparse_vector.get("indices", [])
    values = sparse_vector.get("values", [])
    if len(indices) != len(values):
        min_len = min(len(indices), len(values))
        indices = indices[:min_len]
        values = values[:min_len]

    cleaned = []
    for idx, val in zip(indices, values):
        try:
            idx_int = int(idx)
            val_float = float(val)
        except (ValueError, TypeError):
            continue

        if 0 <= idx_int < SPARSE_DIMENSION and val_float > 0:
            cleaned.append((idx_int, val_float))

    if not cleaned:
        return f"{{}}/{SPARSE_DIMENSION}"

    pairs = [f"{idx}:{val:.6f}" for idx, val in cleaned]
    return "{" + ",".join(pairs) + f"}}/{SPARSE_DIMENSION}"


class DatabaseRepository:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL khong duoc tim thay trong file .env")

    def _get_connection(self):
        return psycopg2.connect(self.database_url)

    def check_job_exists(self, job_id: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM public.job_responsibility_chunks WHERE job_id = %s LIMIT 1",
                (job_id,),
            )
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        finally:
            conn.close()

    def check_cv_exists(self, cv_id: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM public.cv_experience_chunks WHERE cv_id = %s LIMIT 1",
                (cv_id,),
            )
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        finally:
            conn.close()

    def get_responsibility_chunks(self, job_id: str) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
                SELECT id, content, vec::text as vec, idx
                FROM public.job_responsibility_chunks
                WHERE job_id = %s
                  AND vec IS NOT NULL
                  AND content IS NOT NULL
                  AND content != ''
                ORDER BY idx
            """
            cursor.execute(query, (job_id,))
            results = cursor.fetchall()
            cursor.close()

            parsed_results = []
            for row in results:
                row_dict = dict(row)
                row_dict["vec"] = parse_pgvector(row_dict["vec"])
                row_dict["id"] = str(row_dict["id"])
                parsed_results.append(row_dict)

            return parsed_results
        finally:
            conn.close()

    def get_experience_chunks(self, cv_id: str) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
                SELECT id, content, vec::text as vec, idx
                FROM public.cv_experience_chunks
                WHERE cv_id = %s
                  AND vec IS NOT NULL
                  AND content IS NOT NULL
                  AND content != ''
                ORDER BY idx
            """
            cursor.execute(query, (cv_id,))
            results = cursor.fetchall()
            cursor.close()

            parsed_results = []
            for row in results:
                row_dict = dict(row)
                row_dict["vec"] = parse_pgvector(row_dict["vec"])
                row_dict["id"] = str(row_dict["id"])
                parsed_results.append(row_dict)

            return parsed_results
        finally:
            conn.close()

    def get_job_vectors(self, job_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
                SELECT
                    problem_text,
                    capability_text,
                    problem_dense_vector::text as problem_dense_vector,
                    problem_sparse_vector::text as problem_sparse_vector,
                    capability_dense_vector::text as capability_dense_vector,
                    capability_sparse_vector::text as capability_sparse_vector
                FROM public.jobs
                WHERE id = %s
                  AND problem_dense_vector IS NOT NULL
                  AND capability_dense_vector IS NOT NULL
            """
            cursor.execute(query, (job_id,))
            result = cursor.fetchone()
            cursor.close()

            if not result:
                return None

            result_dict = dict(result)
            result_dict["problem_dense_vector"] = parse_pgvector(result_dict["problem_dense_vector"])
            result_dict["capability_dense_vector"] = parse_pgvector(result_dict["capability_dense_vector"])
            result_dict["problem_sparse_vector"] = parse_sparsevec(result_dict["problem_sparse_vector"])
            result_dict["capability_sparse_vector"] = parse_sparsevec(result_dict["capability_sparse_vector"])
            return result_dict
        finally:
            conn.close()

    def get_cv_vectors(self, cv_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
                SELECT
                    problem_text,
                    capability_text,
                    problem_dense_vector::text as problem_dense_vector,
                    problem_sparse_vector::text as problem_sparse_vector,
                    capability_dense_vector::text as capability_dense_vector,
                    capability_sparse_vector::text as capability_sparse_vector
                FROM public.cvs
                WHERE id = %s
                  AND problem_dense_vector IS NOT NULL
                  AND capability_dense_vector IS NOT NULL
            """
            cursor.execute(query, (cv_id,))
            result = cursor.fetchone()
            cursor.close()

            if not result:
                return None

            result_dict = dict(result)
            result_dict["problem_dense_vector"] = parse_pgvector(result_dict["problem_dense_vector"])
            result_dict["capability_dense_vector"] = parse_pgvector(result_dict["capability_dense_vector"])
            result_dict["problem_sparse_vector"] = parse_sparsevec(result_dict["problem_sparse_vector"])
            result_dict["capability_sparse_vector"] = parse_sparsevec(result_dict["capability_sparse_vector"])
            return result_dict
        finally:
            conn.close()

    def find_top_candidates_by_vectors(
        self,
        problem_dense: List[float],
        problem_sparse: Dict,
        capability_dense: List[float],
        capability_sparse: Dict,
        top_k: int = 50,
    ) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            problem_sparse_str = format_sparsevec_query(problem_sparse)
            capability_sparse_str = format_sparsevec_query(capability_sparse)

            query = """
                WITH scores AS (
                    SELECT
                        id as cv_id,
                        problem_text,
                        capability_text,
                        (
                            0.8 * (1 - (problem_dense_vector <=> %s::vector)) +
                            0.2 * (problem_sparse_vector <#> %s::sparsevec)
                        ) as problem_score,
                        (
                            0.5 * (1 - (capability_dense_vector <=> %s::vector)) +
                            0.5 * (capability_sparse_vector <#> %s::sparsevec)
                        ) as capability_score
                    FROM public.cvs
                    WHERE problem_dense_vector IS NOT NULL
                      AND capability_dense_vector IS NOT NULL
                      AND problem_sparse_vector IS NOT NULL
                      AND capability_sparse_vector IS NOT NULL
                )
                SELECT
                    cv_id,
                    problem_text,
                    capability_text,
                    problem_score,
                    capability_score,
                    (0.9 * problem_score + 0.1 * capability_score) as final_score
                FROM scores
                WHERE problem_score >= %s
                ORDER BY final_score DESC
                LIMIT %s
            """

            cursor.execute(
                query,
                (
                    problem_dense,
                    problem_sparse_str,
                    capability_dense,
                    capability_sparse_str,
                    MIN_PROBLEM_SCORE,
                    top_k,
                ),
            )
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        finally:
            conn.close()

    def find_top_jobs_by_vectors(
        self,
        problem_dense: List[float],
        problem_sparse: Dict,
        capability_dense: List[float],
        capability_sparse: Dict,
        top_k: int = 50,
    ) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            problem_sparse_str = format_sparsevec_query(problem_sparse)
            capability_sparse_str = format_sparsevec_query(capability_sparse)

            query = """
                WITH scores AS (
                    SELECT
                        id as job_id,
                        problem_text,
                        capability_text,
                        (
                            0.8 * (1 - (problem_dense_vector <=> %s::vector)) +
                            0.2 * (problem_sparse_vector <#> %s::sparsevec)
                        ) as problem_score,
                        (
                            0.5 * (1 - (capability_dense_vector <=> %s::vector)) +
                            0.5 * (capability_sparse_vector <#> %s::sparsevec)
                        ) as capability_score
                    FROM public.jobs
                    WHERE problem_dense_vector IS NOT NULL
                      AND capability_dense_vector IS NOT NULL
                      AND problem_sparse_vector IS NOT NULL
                      AND capability_sparse_vector IS NOT NULL
                )
                SELECT
                    job_id,
                    problem_text,
                    capability_text,
                    problem_score,
                    capability_score,
                    (0.9 * problem_score + 0.1 * capability_score) as final_score
                FROM scores
                WHERE problem_score >= %s
                ORDER BY final_score DESC
                LIMIT %s
            """

            cursor.execute(
                query,
                (
                    problem_dense,
                    problem_sparse_str,
                    capability_dense,
                    capability_sparse_str,
                    MIN_PROBLEM_SCORE,
                    top_k,
                ),
            )
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        finally:
            conn.close()
