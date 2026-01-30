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
