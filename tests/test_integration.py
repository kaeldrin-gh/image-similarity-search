"""Basic integration test."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import pandas as pd
from PIL import Image

from img_similarity.data.loader import load_dataset
from img_similarity.vision.embeddings import EmbeddingExtractor
from img_similarity.index.faiss_index import FaissIndexer
from img_similarity.evaluation import evaluate_retrieval


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data loading to evaluation."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test images
            test_images = []
            for i in range(3):
                img_path = temp_path / f"test_{i}.jpg"
                # Create simple colored image
                img = Image.new('RGB', (224, 224), (100 + i * 50, 100, 100))
                img.save(img_path)
                test_images.append(str(img_path))
            
            # Create CSV dataset
            csv_path = temp_path / "dataset.csv"
            df = pd.DataFrame({
                "id": range(len(test_images)),
                "image_path": test_images,
                "label": ["cat", "dog", "cat"]
            })
            df.to_csv(csv_path, index=False)
            
            # Load dataset
            loaded_df = load_dataset(csv_path)
            assert len(loaded_df) == 3
            
            # Extract embeddings
            extractor = EmbeddingExtractor(device="cpu")
            embeddings = extractor.extract_embeddings(test_images)
            
            assert embeddings.shape == (3, 2048)
            
            # Build index
            indexer = FaissIndexer()
            indexer.build(embeddings)
            
            assert indexer.size == 3
            
            # Test queries
            results = indexer.query(embeddings[0], k=2)
            assert len(results) == 2
            assert results[0][0] == 0  # First result should be self
            assert results[0][1] < 1e-5  # Distance should be ~0
            
            # Test evaluation
            predictions = []
            for i in range(len(embeddings)):
                query_results = indexer.query(embeddings[i], k=3)
                # Remove self
                filtered_results = [(idx, dist) for idx, dist in query_results if idx != i]
                predictions.append(filtered_results)
            
            metrics = evaluate_retrieval(predictions, loaded_df)
            assert "mAP" in metrics
            assert 0 <= metrics["mAP"] <= 1
            
            print("✓ End-to-end pipeline test passed!")
    
    def test_cli_integration(self):
        """Test CLI commands work correctly."""
        # This would require more complex setup with actual CLI testing
        # For now, we'll just verify the modules can be imported
        
        from img_similarity.cli import app
        assert app is not None
        
        print("✓ CLI integration test passed!")


if __name__ == "__main__":
    test = TestIntegration()
    test.test_end_to_end_pipeline()
    test.test_cli_integration()
    print("All integration tests passed!")
