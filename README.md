# CV - Job Matching AI Service

AI Service dùng để **phân tích và so khớp CV với Job Description** bằng các kỹ thuật **Embedding, Semantic Search và Reranking**.

Service này cung cấp API để:
- Phân tích mức độ phù hợp giữa **CV và Job**
- Tính **matching score**
- Trả về **giải thích chi tiết** về sự phù hợp giữa kinh nghiệm và yêu cầu công việc

Hệ thống được xây dựng với:
- **FastAPI**
- **FlagEmbedding**
- **Supabase / PostgreSQL**
- **Vector Search + Reranking**

---

# 🚀 Getting Started

## 1. Cài đặt thư viện

Cài toàn bộ dependencies cần thiết:

```bash
pip install -r requirements.txt
```

## 2. Chạy AI Service

Chạy server FastAPI:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Sau khi chạy thành công, có thể truy cập:

```bash
http://localhost:8000/docs
```
Swagger UI cho phép test API trực tiếp trên browser, không cần dùng Postman.

# 🧠 Embedding Pipeline

## Hai lệnh dưới đây dùng để tìm các field vector còn trống và tự động sinh embedding .

```bash
python -m app.embedding_pipeline
```

```bash
python -m app.embedding_pipeline
```
# 📌 Project Status

##🚧 Đang trong giai đoạn MVP Development

Các phần đang hoàn thiện thêm:

1.Explain Matching cần tăng độ chính xác

2.Cải thiện retrieve logic 

3.Hoàn thiện công thức cuối cùng để tính điểm hợp lý
