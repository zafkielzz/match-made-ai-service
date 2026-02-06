import os
import torch

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

DEVICE = os.getenv("AI_DEVICE")
if not DEVICE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_FP16 = os.getenv("AI_FP16")
if USE_FP16 is None:
    USE_FP16 = (DEVICE.startswith("cuda"))
else:
    USE_FP16 = (USE_FP16 == "1")

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
