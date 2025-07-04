"""Test configuration and fixtures."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return np.random.rand(10, 2048).astype(np.float32)


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_image_paths(temp_dir):
    """Generate sample image paths for testing."""
    paths = []
    for i in range(5):
        img_path = temp_dir / f"image_{i}.jpg"
        img_path.touch()  # Create empty file
        paths.append(str(img_path))
    return paths
