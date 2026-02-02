# Match-Made-AI – AI Core (Inference Service)

**AI inference service** for the Match-Made-AI platform.

This service provides:
- Dense **text embeddings** for semantic retrieval
- **Reranking / scoring** for job–candidate matching
- **Parsing** raw job descriptions into structured JSON

Built with **FastAPI** and **FlagEmbedding**, exposed via lightweight REST APIs.

> **Status**: Work in progress  
> **Best results** with clean, normalized input text (canonical CV / JD format)

---

## Scope & Responsibilities

### What this service does
- Generate embeddings for text inputs
- Compute semantic similarity / rerank scores
- Parse unstructured job descriptions into structured data
- Serve models via stateless REST APIs
- Remain **model-agnostic** and easy to extend

### What this service does NOT do
- Authentication or authorization
- Data persistence or storage
- Business logic or application rules

These concerns are handled by the backend layer.

---

## Models & Configuration

All runtime configuration is defined in `config.py`:

- `DEVICE` – `cpu` or `cuda`
- `USE_FP16` – FP16 inference on supported GPUs
- `EMBED_MODEL_NAME`
- `RERANK_MODEL_NAME`
- `MAX_LENGTH`
- `EMBED_BATCH_SIZE`
- `RERANK_BATCH_SIZE`

Models are lazily loaded via:
- `get_embedder()`
- `get_reranker()`

---

## API Overview

**Base URL**: `http://localhost:<port>`

| Endpoint | Description |
| :--- | :--- |
| `GET /health` | Health check & runtime config |
| `POST /embed` | Generate embeddings |
| `POST /rerank` | Rerank one query vs many candidates |
| `POST /rerank_pairs` | Rerank explicit query–candidate pairs |
| `POST /jobs/parse` | Parse raw job description text |

---

## API Details

### `GET /health`
Health check and runtime configuration snapshot.

**Response**
```json
{
  "status": "ok",
  "device": "cpu",
  "fp16": false,
  "embed_model": "...",
  "rerank_model": "..."
}
```

### `POST /embed`
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
**Notes:**
- Output vectors are `float32`.
- If `normalize=true`, vectors are L2-normalized using FAISS.

### `POST /rerank`
Compute rerank scores for one query against multiple candidates.

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

### `POST /rerank_pairs`
Compute rerank scores for explicit query–candidate pairs.

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
**Validation:**
- Each item in `pairs` must be `[query, candidate]`.
- Invalid input returns `400`.

### `POST /jobs/parse`
Parse raw job description text into clean, structured JSON optimized for filtering and semantic matching.

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
    "locations": [],
    "jobLevel": "INTERN",
    "employmentType": "INTERNSHIP",
    "experienceYears": { "min": 0, "max": 0 },
    "education": { "minLevel": "BACHELOR" },
    "salary": {
      "min": 0,
      "max": 0,
      "currency": "VND",
      "period": "MONTH",
      "type": "GROSS",
      "negotiable": true
    },
    "overview": "",
    "responsibilities": [],
    "requirements": {
      "required": [],
      "preferred": []
    }
  },
  "meta": {
    "confidence": {},
    "warnings": []
  }
}
```

---

## Intended Workflow

Typical usage flow:

1.  **Parse raw JD** → structured JSON
2.  **Convert structured data** → canonical text
3.  **Generate embeddings**
4.  **Rerank candidates**

> **Note:** Canonical text generation should be handled by the backend to ensure consistency.

---

## Data Assumptions

This service assumes:
- Input text is clean and normalized.
- Job and CV data follow a consistent schema.
- Free-text noise reduces matching quality.

---

## Tech Stack

* **Framework:** FastAPI
* **Language:** Python
* **Models:** FlagEmbedding
* **Vector ops:** FAISS (L2 normalization only)
* **Serving:** REST API

---

## Run Locally

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**3. Health check:**
```bash
curl http://localhost:8000/health
```

---

## Limitations & Roadmap

### Current
- No caching or persistence
- Rerank outputs are raw scores only
- Matching quality depends on input normalization

### Planned
- Canonical CV/JD formatter
- Hybrid scoring (semantic + rule-based)
- Batch endpoints and caching
- Explainability metadata

---

## License
Educational / experimental use.
