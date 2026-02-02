import os

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

DEVICE = os.getenv("AI_DEVICE", "cpu")      # "cpu" | "cuda"
USE_FP16 = os.getenv("AI_FP16", "0") == "1"

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
