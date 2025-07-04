"""Test data loading functionality."""

import pandas as pd
import pytest
from pathlib import Path

from img_similarity.data.loader import load_dataset


class TestDataLoader:
    """Test cases for data loading functionality."""
    
    def test_load_csv_dataset(self, temp_dir):
        """Test loading dataset from CSV file."""
        # Create test CSV
        csv_path = temp_dir / "test_data.csv"
        test_data = pd.DataFrame({
            "id": [1, 2, 3],
            "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "label": ["cat", "dog", "cat"]
        })
        test_data.to_csv(csv_path, index=False)
        
        # Load dataset
        df = load_dataset(csv_path)
        
        # Verify results
        assert len(df) == 3
        assert "image_path" in df.columns
        assert "label" in df.columns
        assert "id" in df.columns
        assert df["label"].tolist() == ["cat", "dog", "cat"]
    
    def test_load_csv_missing_column(self, temp_dir):
        """Test loading CSV with missing image column."""
        csv_path = temp_dir / "test_data.csv"
        test_data = pd.DataFrame({
            "id": [1, 2, 3],
            "wrong_col": ["img1.jpg", "img2.jpg", "img3.jpg"]
        })
        test_data.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Image column.*not found"):
            load_dataset(csv_path)
    
    def test_load_directory_dataset(self, temp_dir):
        """Test loading dataset from directory structure."""
        # Create test directory structure
        cat_dir = temp_dir / "cats"
        dog_dir = temp_dir / "dogs"
        cat_dir.mkdir()
        dog_dir.mkdir()
        
        # Create test image files
        (cat_dir / "cat1.jpg").touch()
        (cat_dir / "cat2.png").touch()
        (dog_dir / "dog1.jpg").touch()
        
        # Load dataset
        df = load_dataset(temp_dir)
        
        # Verify results - should have at least 3 images
        assert len(df) >= 3
        assert "image_path" in df.columns
        assert "label" in df.columns
        assert "id" in df.columns
        
        # Check that labels are assigned correctly
        labels = df["label"].tolist()
        assert "cats" in labels
        assert "dogs" in labels
    
    def test_load_nonexistent_path(self):
        """Test loading from non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path")
    
    def test_load_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        with pytest.raises(ValueError, match="No image files found"):
            load_dataset(temp_dir)
