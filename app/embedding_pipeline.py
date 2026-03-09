"""
Pipeline để tự động tạo embeddings cho các bảng cv_experience_chunks và job_responsibility_chunks.
Xử lý các row chưa có vector (vec IS NULL), gọi embed_texts và cập nhật vào database.
"""

import os
import sys
from typing import List, Tuple
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Import hàm embed_texts từ models
from .models import embed_texts

load_dotenv()


def get_db_connection():
    """Tạo kết nối đến PostgreSQL database từ DATABASE_URL trong .env"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL không được tìm thấy trong file .env")
    
    try:
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        raise ConnectionError(f"Không thể kết nối database: {e}")


def fetch_rows_without_vectors(conn, table_name: str, batch_size: int = 100) -> List[Tuple]:
    """
    Lấy các row chưa có vector (vec IS NULL) từ bảng.
    
    Args:
        conn: Database connection
        table_name: Tên bảng (vd: 'public.cv_experience_chunks')
        batch_size: Số lượng row lấy mỗi lần
    
    Returns:
        List of tuples (id, content)
    """
    cursor = conn.cursor()
    query = f"""
        SELECT id, content 
        FROM {table_name} 
        WHERE vec IS NULL AND content IS NOT NULL AND content != ''
        LIMIT %s
    """
    cursor.execute(query, (batch_size,))
    rows = cursor.fetchall()
    cursor.close()
    return rows


def update_vectors(conn, table_name: str, updates: List[Tuple[List[float], int]]):
    """
    Cập nhật vectors vào cột vec của bảng.
    
    Args:
        conn: Database connection
        table_name: Tên bảng
        updates: List of tuples (vector, id)
    """
    cursor = conn.cursor()
    query = f"""
        UPDATE {table_name}
        SET vec = %s
        WHERE id = %s
    """
    execute_batch(cursor, query, updates)
    conn.commit()
    cursor.close()


def process_table(conn, table_name: str, batch_size: int = 100) -> int:
    """
    Xử lý một bảng: lấy rows chưa có vector, tạo embeddings, và cập nhật.
    
    Args:
        conn: Database connection
        table_name: Tên bảng đầy đủ (vd: 'public.cv_experience_chunks')
        batch_size: Số lượng row xử lý mỗi batch
    
    Returns:
        Tổng số row đã xử lý
    """
    total_processed = 0
    
    print(f"\n{'='*60}")
    print(f"Đang xử lý bảng: {table_name}")
    print(f"{'='*60}")
    
    while True:
        # Lấy batch rows chưa có vector
        rows = fetch_rows_without_vectors(conn, table_name, batch_size)
        
        if not rows:
            print(f"✓ Hoàn thành! Không còn row nào cần xử lý.")
            break
        
        print(f"\nĐang xử lý {len(rows)} rows...")
        
        # Tách id và content
        ids = [row[0] for row in rows]
        contents = [row[1] for row in rows]
        
        # Gọi embed_texts để tạo vectors
        try:
            vectors = embed_texts(contents)
            print(f"✓ Đã tạo {len(vectors)} embeddings")
        except Exception as e:
            print(f"✗ Lỗi khi tạo embeddings: {e}")
            continue
        
        # Chuẩn bị dữ liệu để update
        updates = [(vector, id_val) for vector, id_val in zip(vectors, ids)]
        
        # Cập nhật vào database
        try:
            update_vectors(conn, table_name, updates)
            total_processed += len(updates)
            print(f"✓ Đã cập nhật {len(updates)} vectors vào database")
            print(f"  Tổng đã xử lý: {total_processed} rows")
        except Exception as e:
            print(f"✗ Lỗi khi cập nhật database: {e}")
            conn.rollback()
            continue
    
    return total_processed


def run_pipeline(batch_size: int = 100):
    """
    Chạy pipeline cho cả 2 bảng cv_experience_chunks và job_responsibility_chunks.
    
    Args:
        batch_size: Số lượng row xử lý mỗi batch (default: 100)
    """
    tables = [
        "public.cv_experience_chunks",
        "public.job_responsibility_chunks"
    ]
    
    print("\n" + "="*60)
    print("BẮT ĐẦU EMBEDDING PIPELINE")
    print("="*60)
    
    try:
        # Kết nối database
        print("\n→ Đang kết nối database...")
        conn = get_db_connection()
        print("✓ Kết nối thành công!")
        
        total_all = 0
        
        # Xử lý từng bảng
        for table in tables:
            try:
                processed = process_table(conn, table, batch_size)
                total_all += processed
            except Exception as e:
                print(f"\n✗ Lỗi khi xử lý bảng {table}: {e}")
                continue
        
        # Đóng kết nối
        conn.close()
        
        print("\n" + "="*60)
        print(f"HOÀN THÀNH! Tổng cộng đã xử lý: {total_all} rows")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Lỗi nghiêm trọng: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Có thể điều chỉnh batch_size tùy theo GPU memory
    # Batch size càng lớn thì xử lý càng nhanh nhưng tốn nhiều memory hơn
    run_pipeline(batch_size=100)
