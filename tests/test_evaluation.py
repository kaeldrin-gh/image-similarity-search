"""Test evaluation functionality."""

import pandas as pd
import pytest
from typing import List, Tuple

from img_similarity.evaluation import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_average_precision,
    evaluate_retrieval,
)


class TestEvaluation:
    """Test cases for evaluation functionality."""
    
    def test_compute_recall_at_k(self):
        """Test Recall@k computation."""
        # Create test data
        labels_df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "label": ["cat", "dog", "cat", "dog", "cat"]
        })
        
        # Test case: query is cat (index 0), predictions include other cats
        predicted_indices = [1, 2, 3]  # dog, cat, dog
        recall = compute_recall_at_k(
            [], predicted_indices, k=3, labels_df=labels_df, query_idx=0
        )
        
        # Should find 1 relevant item (index 2) out of 2 total relevant items
        expected_recall = 1.0 / 2.0  # 1 found / 2 total cats (excluding query)
        assert recall == expected_recall
    
    def test_compute_precision_at_k(self):
        """Test Precision@k computation."""
        labels_df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "label": ["cat", "dog", "cat", "dog", "cat"]
        })
        
        # Test case: query is cat (index 0), predictions include other cats
        predicted_indices = [1, 2, 3]  # dog, cat, dog
        precision = compute_precision_at_k(
            predicted_indices, k=3, labels_df=labels_df, query_idx=0
        )
        
        # Should find 1 relevant item out of 3 predictions
        expected_precision = 1.0 / 3.0
        assert precision == expected_precision
    
    def test_compute_average_precision(self):
        """Test Average Precision computation."""
        labels_df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "label": ["cat", "dog", "cat", "dog", "cat"]
        })
        
        # Test case: query is cat (index 0)
        predicted_indices = [2, 1, 4]  # cat, dog, cat
        ap = compute_average_precision(
            predicted_indices, labels_df=labels_df, query_idx=0, max_k=3
        )
        
        # Relevant items at positions 1 and 3
        # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        expected_ap = (1.0 + 2.0/3.0) / 2.0
        assert abs(ap - expected_ap) < 1e-6
    
    def test_evaluate_retrieval(self):
        """Test complete retrieval evaluation."""
        # Create test data
        labels_df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "label": ["cat", "dog", "cat", "dog", "cat"]
        })
        
        # Create predictions for 2 queries
        predictions = [
            [(2, 0.1), (1, 0.2), (4, 0.3)],  # Query 0 (cat): finds cat, dog, cat
            [(3, 0.1), (0, 0.2), (2, 0.3)],  # Query 1 (dog): finds dog, cat, cat
        ]
        
        metrics = evaluate_retrieval(predictions, labels_df, k_values=[1, 3])
        
        # Check that metrics are computed
        assert "Recall@1" in metrics
        assert "Precision@1" in metrics
        assert "Recall@3" in metrics
        assert "Precision@3" in metrics
        assert "mAP" in metrics
        assert "mAP@1" in metrics
        assert "mAP@3" in metrics
        
        # Check that values are reasonable
        assert 0 <= metrics["Recall@1"] <= 1
        assert 0 <= metrics["Precision@1"] <= 1
        assert 0 <= metrics["mAP"] <= 1
    
    def test_evaluate_empty_predictions(self):
        """Test evaluation with empty predictions."""
        labels_df = pd.DataFrame({
            "id": [0, 1, 2],
            "label": ["cat", "dog", "cat"]
        })
        
        predictions = []
        metrics = evaluate_retrieval(predictions, labels_df)
        
        # Should return empty metrics
        assert metrics == {}
    
    def test_evaluate_no_labels(self):
        """Test evaluation with no prediction results."""
        labels_df = pd.DataFrame({
            "id": [0, 1, 2],
            "label": ["cat", "dog", "cat"]
        })
        
        predictions = [[], []]  # Empty predictions
        metrics = evaluate_retrieval(predictions, labels_df)
        
        # Should return empty metrics
        assert metrics == {}
