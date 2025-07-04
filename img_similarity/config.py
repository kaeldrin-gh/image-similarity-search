"""Configuration settings for the image similarity search system."""

import os
from pathlib import Path
from typing import Optional

# Random seeds for reproducibility
RANDOM_SEED = 42
TORCH_SEED = 42
NUMPY_SEED = 42

# Model configuration
DEFAULT_MODEL_NAME = "resnet50"
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 224
EMBEDDING_DIM = 2048

# Index configuration
DEFAULT_INDEX_TYPE = "faiss"
DEFAULT_TOP_K = 10

# Evaluation configuration
EVAL_K_VALUES = [1, 5, 10]

# File paths
DATA_DIR = Path("data")
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "faiss.idx"
LABELS_FILE = "labels.csv"

# Environment variables
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Dataset configuration
DEFAULT_IMG_COL = "image_path"
DEFAULT_LABEL_COL = "label"
DEFAULT_ID_COL = "id"
