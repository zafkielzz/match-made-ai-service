"""
Pipeline de tu dong tao multi-vector embeddings cho bang jobs va cvs.
Xu ly cac row chua co vectors, goi embed_texts_hybrid va cap nhat vao database.

Moi job/cv se co 4 vectors:
- problem_dense_vector: Dense embedding cho problem space
- problem_sparse_vector: Sparse embedding cho problem space
- capability_dense_vector: Dense embedding cho capability space
- capability_sparse_vector: Sparse embedding cho capability space
"""

import json
import os
import sys
from typing import List, Tuple

import psycopg2
from dotenv import load_dotenv

from .models import embed_texts_hybrid

load_dotenv()

SPARSE_DIMENSION = 30522


def check_column_types(conn, table_name: str):
    """
    Kiem tra data type cua cac vector columns trong bang.
    """
    cursor = conn.cursor()
    query = """
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_name=%s AND column_name LIKE '%%vector%%'
        ORDER BY column_name
    """
    cursor.execute(query, (table_name.split(".")[-1],))
    results = cursor.fetchall()
    cursor.close()

    print(f"\n--- Column Types trong {table_name} ---")
    for col_name, data_type, udt_name in results:
        print(f"  {col_name}: {data_type} (udt_name={udt_name})")

    return results


def get_db_connection():
    """Tao ket noi den PostgreSQL database tu DATABASE_URL trong .env"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL khong duoc tim thay trong file .env")

    try:
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        raise ConnectionError(f"Khong the ket noi database: {e}")


def format_sparse_vector_for_db(sparse_dict, use_json: bool = False) -> str:
    """
    Convert sparse vector dict de luu vao PostgreSQL.

    Args:
        sparse_dict: Dict or JSON string
                    Format: {"indices": [1,2,3], "values": [0.5,0.8,0.3]}
        use_json: If True, return JSON string; if False, return sparsevec text format

    Returns:
        - JSON format: '{"indices":[1,2,3],"values":[0.5,0.8,0.3]}'
        - Sparsevec text format: '{1:0.5,2:0.8}/30522'
    """
    try:
        if isinstance(sparse_dict, str):
            sparse_dict = json.loads(sparse_dict)

        sparse_dict = sparse_dict or {}
        indices = sparse_dict.get("indices", [])
        values = sparse_dict.get("values", [])

        if len(indices) != len(values):
            min_len = min(len(indices), len(values))
            indices = indices[:min_len]
            values = values[:min_len]

        cleaned_indices = []
        cleaned_values = []
        for idx, val in zip(indices, values):
            try:
                idx_int = int(idx)
                val_float = float(val)
            except (ValueError, TypeError):
                continue

            if 0 <= idx_int < SPARSE_DIMENSION and val_float > 0:
                cleaned_indices.append(idx_int)
                cleaned_values.append(round(val_float, 6))

        if use_json:
            return json.dumps({
                "indices": cleaned_indices,
                "values": cleaned_values,
            })

        if not cleaned_indices:
            return f"{{}}/{SPARSE_DIMENSION}"

        pairs = [f"{idx}:{val:.6f}" for idx, val in zip(cleaned_indices, cleaned_values)]
        return "{" + ",".join(pairs) + f"}}/{SPARSE_DIMENSION}"
    except Exception as e:
        print(f"X Error khi format sparse vector: {e}")
        if use_json:
            return json.dumps({"indices": [], "values": []})
        return f"{{}}/{SPARSE_DIMENSION}"



def boost_problem_text(title: str, problem_text: str) -> str:
    """
    Boost title signal in problem_text before embedding.

    Note:
    - DB currently exposes `title` but not a dedicated `extracted_skills` column.
    - We repeat `title` twice at the top and keep the original problem_text unchanged after it.
    """
    title = (title or "").strip()
    problem_text = (problem_text or "").strip()

    boosted_parts = []
    if title:
        boosted_parts.extend([title, title])
    if problem_text:
        boosted_parts.append(problem_text)

    return "\n".join(part for part in boosted_parts if part)
def fetch_rows_without_vectors(conn, table_name: str, batch_size: int = 50) -> List[Tuple]:
    """
    Lay cac row chua co multi-vectors tu bang jobs hoac cvs.
    """
    cursor = conn.cursor()
    query = f"""
        SELECT id, title, problem_text, capability_text
        FROM {table_name}
        WHERE (
            problem_dense_vector IS NULL
            OR problem_sparse_vector IS NULL
            OR capability_dense_vector IS NULL
            OR capability_sparse_vector IS NULL
        )
        AND problem_text IS NOT NULL
        AND problem_text != ''
        AND capability_text IS NOT NULL
        AND capability_text != ''
        LIMIT %s
    """
    cursor.execute(query, (batch_size,))
    rows = cursor.fetchall()
    cursor.close()
    return rows


def update_multi_vectors(
    conn,
    table_name: str,
    updates: List[Tuple],
    use_json: bool = False,
) -> int:
    """
    Cap nhat 4 vectors vao bang.

    updates: List[(problem_dense, problem_sparse, capability_dense, capability_sparse, id)]
    """
    cursor = conn.cursor()

    if use_json:
        query = f"""
            UPDATE {table_name}
            SET
                problem_dense_vector = %s::vector,
                problem_sparse_vector = %s::jsonb,
                capability_dense_vector = %s::vector,
                capability_sparse_vector = %s::jsonb
            WHERE id = %s
        """
    else:
        query = f"""
            UPDATE {table_name}
            SET
                problem_dense_vector = %s::vector,
                problem_sparse_vector = %s::sparsevec,
                capability_dense_vector = %s::vector,
                capability_sparse_vector = %s::sparsevec
            WHERE id = %s
        """

    successful = 0
    had_error = False

    for i, update_data in enumerate(updates):
        record_id = update_data[-1] if update_data else "<missing-id>"
        if len(update_data) != 5:
            had_error = True
            print(f"  X Loi cap nhat record {i + 1}/{len(updates)}: ID={record_id}")
            print(f"    Error: update_data phai co 5 phan tu nhung nhan {len(update_data)}")
            continue

        savepoint_name = f"multi_vector_row_{i}"
        cursor.execute(f"SAVEPOINT {savepoint_name}")

        try:
            cursor.execute(query, update_data)
            if cursor.rowcount != 1:
                raise ValueError(f"Khong tim thay row can update (rowcount={cursor.rowcount})")
            cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            successful += 1
        except Exception as e:
            had_error = True
            cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            print(f"  X Loi cap nhat record {i + 1}/{len(updates)}: ID={record_id}")
            print(f"    Error: {e}")
            print(
                f"    Problem sparse (type={type(update_data[1])}, len={len(str(update_data[1]))}): "
                f"{str(update_data[1])[:120]}..."
            )
            print(
                f"    Capability sparse (type={type(update_data[3])}, len={len(str(update_data[3]))}): "
                f"{str(update_data[3])[:120]}..."
            )

    if successful > 0:
        conn.commit()
    elif had_error:
        conn.rollback()

    cursor.close()
    return successful


def process_table(conn, table_name: str, batch_size: int = 50) -> int:
    """
    Xu ly mot bang: lay rows chua co vectors, tao embeddings, va cap nhat.
    """
    total_processed = 0

    print(f"\n{'=' * 60}")
    print(f"Dang xu ly bang: {table_name}")
    print(f"{'=' * 60}")

    col_types = check_column_types(conn, table_name)

    use_json = False
    if col_types:
        for _, data_type, udt_name in col_types:
            if "jsonb" in data_type.lower() or "json" in data_type.lower():
                use_json = True
                print("  -> Detected JSONB column, will use JSON format for sparse vectors")
                break
            if udt_name == "sparsevec":
                use_json = False
                print("  -> Detected sparsevec UDT, will use sparsevec format")
                break

    print(f"  Format: {'JSON' if use_json else 'Sparsevec text'}\n")

    while True:
        rows = fetch_rows_without_vectors(conn, table_name, batch_size)

        if not rows:
            print("Hoan thanh! Khong con row nao can xu ly.")
            break

        print(f"\nDang xu ly {len(rows)} rows...")

        ids = [row[0] for row in rows]
        problem_texts = [row[1] for row in rows]
        capability_texts = [row[2] for row in rows]

        try:
            print("  -> Tao embeddings cho Problem Space...")
            problem_results = embed_texts_hybrid(problem_texts)
            print(f"  OK Da tao {len(problem_results)} problem embeddings")
        except Exception as e:
            print(f"  X Loi khi tao problem embeddings: {e}")
            continue

        try:
            print("  -> Tao embeddings cho Capability Space...")
            capability_results = embed_texts_hybrid(capability_texts)
            print(f"  OK Da tao {len(capability_results)} capability embeddings")
        except Exception as e:
            print(f"  X Loi khi tao capability embeddings: {e}")
            continue

        if len(problem_results) != len(ids) or len(capability_results) != len(ids):
            print(
                "  X So luong embeddings khong khop voi so rows: "
                f"rows={len(ids)}, problem={len(problem_results)}, capability={len(capability_results)}"
            )
            continue

        updates = []
        for i, id_val in enumerate(ids):
            updates.append(
                (
                    problem_results[i]["dense_vector"],
                    format_sparse_vector_for_db(problem_results[i]["sparse_vector"], use_json=use_json),
                    capability_results[i]["dense_vector"],
                    format_sparse_vector_for_db(capability_results[i]["sparse_vector"], use_json=use_json),
                    id_val,
                )
            )

        try:
            print("  -> Cap nhat vao database...")
            successful = update_multi_vectors(conn, table_name, updates, use_json=use_json)
            total_processed += successful
            if successful == len(updates):
                print(f"  OK Da cap nhat {successful}/{len(updates)} rows")
            else:
                print(f"  Canh bao Chi cap nhat {successful}/{len(updates)} rows")
            print(f"  OK Tong da xu ly: {total_processed} rows")
        except Exception as e:
            print(f"  X Loi khi cap nhat database: {e}")
            conn.rollback()
            continue

    return total_processed


def run_pipeline(batch_size: int = 50):
    """
    Chay pipeline cho ca 2 bang jobs va cvs.
    """
    tables = ["public.jobs", "public.cvs"]

    print("\n" + "=" * 60)
    print("BAT DAU MULTI-VECTOR EMBEDDING PIPELINE")
    print("=" * 60)
    print("\nMoi record se co 4 vectors:")
    print("  1. Problem Dense Vector (1024-dim)")
    print("  2. Problem Sparse Vector (lexical weights)")
    print("  3. Capability Dense Vector (1024-dim)")
    print("  4. Capability Sparse Vector (lexical weights)")

    try:
        print("\n-> Dang ket noi database...")
        conn = get_db_connection()
        print("OK Ket noi thanh cong!")

        total_all = 0
        for table in tables:
            try:
                processed = process_table(conn, table, batch_size)
                total_all += processed
            except Exception as e:
                print(f"\nX Loi khi xu ly bang {table}: {e}")
                continue

        conn.close()

        print("\n" + "=" * 60)
        print(f"HOAN THANH! Tong cong da xu ly: {total_all} rows")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\nX Loi nghiem trong: {e}")
        sys.exit(1)


def check_status():
    """
    Kiem tra trang thai embedding cua cac bang.
    """
    print("\n" + "=" * 60)
    print("KIEM TRA TRANG THAI EMBEDDING")
    print("=" * 60)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        tables = ["public.jobs", "public.cvs"]

        for table in tables:
            print(f"\n--- {table.upper()} ---")

            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total = cursor.fetchone()[0]
            print(f"  Tong so rows: {total}")

            cursor.execute(
                f"""
                SELECT COUNT(*) FROM {table}
                WHERE problem_dense_vector IS NOT NULL
                  AND problem_sparse_vector IS NOT NULL
                  AND capability_dense_vector IS NOT NULL
                  AND capability_sparse_vector IS NOT NULL
            """
            )
            completed = cursor.fetchone()[0]
            completed_pct = (completed / total * 100) if total else 0
            print(f"  Da co vectors: {completed} ({completed_pct:.1f}%)")

            pending = total - completed
            pending_pct = (pending / total * 100) if total else 0
            print(f"  Chua co vectors: {pending} ({pending_pct:.1f}%)")

            cursor.execute(
                f"""
                SELECT COUNT(*) FROM {table}
                WHERE problem_text IS NULL
                   OR problem_text = ''
                   OR capability_text IS NULL
                   OR capability_text = ''
            """
            )
            missing_text = cursor.fetchone()[0]
            if missing_text > 0:
                print(f"  Canh bao Thieu text: {missing_text} rows (khong the embed)")

        cursor.close()
        conn.close()

        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"\nX Loi khi kiem tra status: {e}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Vector Embedding Pipeline cho Jobs va CVs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="So luong rows xu ly moi batch (default: 50)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Chi kiem tra trang thai, khong chay pipeline",
    )

    args = parser.parse_args()

    if args.status:
        check_status()
    else:
        run_pipeline(batch_size=args.batch_size)


