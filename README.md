# Match-Made-AI – AI Core (FastAPI Inference Service)

AI Core inference service for **Match-Made-AI**, providing:
- **Embedding** for semantic retrieval
- **Reranking / scoring** for job–candidate matching

This service uses **FlagEmbedding** models and is exposed via REST APIs using **FastAPI**.

>  Status: **Work in progress**  
>  Performs best with **clean, normalized input text** (ideally following the same schema as frontend/backend).

---

##  Responsibilities

- Generate dense embeddings for text inputs
- Compute semantic similarity scores between queries and candidates
- Serve AI inference via lightweight REST APIs
- Remain **model-agnostic** and easy to replace or extend
- Parse raw job description text into clean, structured JSON, ready for matching

This service **does NOT** handle:
- Authentication
- Persistence
- Business rules

These concerns are handled by the backend layer.

---

##  Models & Configuration

Configured in `config.py`:

- `DEVICE` – inference device (`cpu` / `cuda`)
- `USE_FP16` – enable FP16 on supported GPUs
- `EMBED_MODEL_NAME` – FlagEmbedding embedding model
- `RERANK_MODEL_NAME` – FlagEmbedding rerank model
- `MAX_LENGTH` – max token length
- `EMBED_BATCH_SIZE`
- `RERANK_BATCH_SIZE`

Model loading is handled via:
- `get_embedder()`
- `get_reranker()`

---

##  API Endpoints

Base URL: `http://localhost:<port>`

---

### GET /health

Health check and runtime configuration snapshot.

**Response**
```json
{
  "status": "ok",
  "device": "cpu",
  "fp16": false,
  "embed_model": "…",
  "rerank_model": "…"
}
```

---

### POST /embed

Generate embeddings for a list of texts.

**Request**
```json
{
  "texts": [
    "Backend developer with Java, REST API, SQL",
    "Frontend developer with React and JavaScript"
  ],
  "normalize": true
}
```

**Response**
```json
{
  "vectors": [
    [0.01, -0.02, 0.03],
    [0.03, 0.04, -0.01]
  ],
  "dim": 768
}
```

**Notes**
- Output vectors are `float32`
- If `normalize=true`, vectors are L2-normalized using FAISS

---

### POST /rerank

Compute rerank scores for **one query** against multiple candidates.

**Request**
```json
{
  "query": "Backend developer with Java and REST API experience",
  "candidates": [
    "Junior backend developer skilled in Java, Spring Boot, SQL",
    "Frontend developer with React and UI focus"
  ],
  "normalize": true
}
```

**Response**
```json
{
  "scores": [0.82, 0.34]
}
```

---

### POST /rerank_pairs

Compute rerank scores for **explicit query–candidate pairs**.

**Request**
```json
{
  "pairs": [
    ["Backend developer with Java", "Java Spring backend engineer"],
    ["Data analyst", "Frontend React developer"]
  ],
  "normalize": true
}
```

**Response**
```json
{
  "scores": [0.90, 0.21]
}
```

**Validation**
- Each item in `pairs` must be `[query, candidate]`
- Invalid input returns `400`

---

### POST /jobs/parse

Parse unstructured job description in to **clean JSON optimized for filtering and semantic matching.**.

**Request**
```json
{
  "rawText": "full job description"
}
```

**Response**
```json
{
  "suggested": {
    "title": "",
    "companyName": null,
    "locations": [
      {
        "country": {
          "code": "VN",
          "name": "Vietnam"
        },
        "city": {
          "code": "VN-HN",
          "name": "Hà Nội"
        },
        "address": null,
        "workModeHint": null
      }
    ],
    "taxonomy": null,
    "jobLevel": "INTERN",
    "employmentType": "INTERNSHIP",
    "workMode": null,
    "experienceYears": {
      "min": 0,
      "max": 0
    },
    "education": {
      "minLevel": "BACHELOR"
    },
    "languageRequirements": [],
    "salary": {
      "min": 0,
      "max": 0,
      "currency": "VND",
      "period": "MONTH",
      "type": "GROSS",
      "negotiable": true
    },
    "overview": "Vị trí thực tập sinh AI Engineer Intern (Computer Vision). Nhiệm vụ chính bao gồm nắm, tham, ứng và các công việc liên quan.",
    "responsibilities": [
      "Nắm vững nền tảng Machine Learning, Deep Learning và các kiến trúc hiện đại.",
      "Tham gia, nghiên cứu và triển khai dự án trong AI Camera, chuyển đổi số và ứng dụng đa lĩnh vực.",
      "Ứng dụng công nghệ mới: Generative AI, Multimodal AI, Edge AI, Agentic AI systems để nâng cao chất lượng sản phẩm.",
      "Góp mặt trong quá trình thiết kế, xây dựng và tối ưu giải pháp AI thực tiễn.",
      "Có khả năng tự nghiên cứu, sáng tạo, phát triển giải pháp đột phá theo xu hướng.",
      "Trở thành kỹ sư AI toàn diện: giỏi chuyên môn, vững kỹ năng mềm, sẵn sàng dẫn dắt dự án."
    ],
    "requirements": {
      "required": [
        "Sinh viên năm cuối đại học các ngành Khoa học máy tính, Công nghệ thông tin, Toán tin,..",
        "Có khả năng làm việc fulltime hoặc parttime tối thiểu 70% số buổi/tuần( Hoặc tối thiểu 28 giờ/ tuần)",
        "Điểm trung bình tích lũy từ 3.2/4.0 trở lên",
        "Tiếng Anh khá, tương đương TOEIC từ 600 trở lên",
        "Có khả năng tự học và chủ động cao trong công việc, yêu thích công nghệ."
      ],
      "preferred": [
        "Ưu tiên các bạn có các dự án cá nhân, giải thưởng hoặc từng tham gia các cuộc thi nghiên cứu, sáng tạo."
      ]
    },
    "benefits": {
      "predefined": [],
      "custom": []
    },
    "workingTime": "Thứ 2–Thứ 6, 08:00–17:00",
    "applicationDeadline": "2026-03-01T00:00:00Z",
    "hireNumber": 1,
    "status": "DRAFT"
  },
  "meta": {
    "confidence": {
      "hireNumber": 0.85,
      "experienceYears": 0.9,
      "title": 0.95,
      "locations": 0.9,
      "salary": 0.9,
      "jobLevel": 0.85,
      "education.minLevel": 0.6,
      "applicationDeadline": 0.9,
      "workingTime": 0.75,
      "workingTime.structured": {
        "days": [
          "MON",
          "TUE",
          "WED",
          "THU",
          "FRI"
        ],
        "startTime": "08:00",
        "endTime": "17:00"
      },
      "responsibilities": 0.75,
      "requirements.required": 0.75,
      "requirements.preferred": 0.6,
      "benefits.custom": 0,
      "employmentType": 0.8,
      "employmentTypeHint": "INTERNSHIP",
      "overview": 0.7
    },
    "warnings": [
      "Salary is negotiable; min/max unknown",
      "Benefits section detected but extraction failed (< 3 items)",
      "Could not confidently detect workMode (insufficient evidence)",
      "INTERN role detected, employmentType set to INTERNSHIP"
    ]
  }
}

```

---

##  Data Assumptions

This service assumes:
- Input text is clean and normalized
- Job/CV data follows a consistent schema
- Free-text noise reduces matching quality

> Recommendation: use a backend formatter to convert structured CV/JD JSON into a canonical text format before inference.

---

##  Tech Stack

- **Framework**: FastAPI
- **Language**: Python
- **Models**: FlagEmbedding
- **Vector ops**: FAISS (L2 normalization only)
- **Serving**: REST API

---

##  Run Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Start server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

---



##  Limitations & Roadmap

**Current**
- No caching or persistence
- Matching quality depends on input normalization
- Rerank outputs are raw scores only

**Planned**
- Canonical CV/JD formatter
- Hybrid scoring (semantic + rules)
- Batch endpoints and caching
- Explainability metadata

---

## 📄 License

Educational / experimental use.
