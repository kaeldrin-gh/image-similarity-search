#!/usr/bin/env python3
"""
Validation script for Image Similarity Search System
Checks that all components are working correctly
"""

import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    try:
        from img_similarity.data.loader import load_dataset
        from img_similarity.vision.embeddings import EmbeddingExtractor  
        from img_similarity.index.faiss_index import FaissIndexer
        from img_similarity.evaluation import evaluate_retrieval
        from img_similarity.visualization import display_query_results
        from img_similarity.cli import app
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with minimal data."""
    print("\nTesting basic functionality...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test images
            test_images = []
            for i in range(3):
                img_path = temp_path / f"test_{i}.jpg"
                img = Image.new('RGB', (224, 224), (100 + i * 50, 100, 100))
                img.save(img_path)
                test_images.append(str(img_path))
            
            # Test data loading
            from img_similarity.data.loader import load_dataset
            csv_path = temp_path / "dataset.csv"
            df = pd.DataFrame({
                "id": range(len(test_images)),
                "image_path": test_images,
                "label": ["cat", "dog", "cat"]
            })
            df.to_csv(csv_path, index=False)
            
            loaded_df = load_dataset(csv_path)
            assert len(loaded_df) == 3
            print("✓ Data loading works")
            
            # Test embedding extraction
            from img_similarity.vision.embeddings import EmbeddingExtractor
            extractor = EmbeddingExtractor(device="cpu")
            embeddings = extractor.extract_embeddings(test_images)
            assert embeddings.shape == (3, 2048)
            print("✓ Embedding extraction works")
            
            # Test indexing
            from img_similarity.index.faiss_index import FaissIndexer
            indexer = FaissIndexer()
            indexer.build(embeddings)
            assert indexer.size == 3
            print("✓ Index building works")
            
            # Test querying
            results = indexer.query(embeddings[0], k=2)
            assert len(results) == 2
            assert results[0][0] == 0  # First result should be self
            print("✓ Query processing works")
            
            # Test evaluation
            from img_similarity.evaluation import evaluate_retrieval
            predictions = []
            for i in range(len(embeddings)):
                query_results = indexer.query(embeddings[i], k=3)
                filtered_results = [(idx, dist) for idx, dist in query_results if idx != i]
                predictions.append(filtered_results)
            
            metrics = evaluate_retrieval(predictions, loaded_df)
            assert "mAP" in metrics
            print("✓ Evaluation works")
            
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Image Similarity Search System Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed! System is ready for use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
