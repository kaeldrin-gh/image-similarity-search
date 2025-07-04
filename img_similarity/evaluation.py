"""Evaluation metrics for image similarity search."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import EVAL_K_VALUES


def compute_recall_at_k(
    true_labels: List[str],
    predicted_indices: List[List[int]],
    k: int,
    labels_df: pd.DataFrame,
    query_idx: int,
) -> float:
    """Compute Recall@k for a single query.
    
    Args:
        true_labels: True labels for the query
        predicted_indices: List of predicted indices for each k
        k: Number of top predictions to consider
        labels_df: DataFrame with labels for all images
        query_idx: Index of the query image
        
    Returns:
        Recall@k score
    """
    if len(predicted_indices) < k:
        k = len(predicted_indices)
    
    # Get labels for predicted indices
    predicted_labels = []
    for idx in predicted_indices[:k]:
        if idx < len(labels_df):
            predicted_labels.append(labels_df.iloc[idx]["label"])
        else:
            predicted_labels.append("unknown")
    
    # Get true label for query
    query_label = labels_df.iloc[query_idx]["label"]
    
    # Count relevant items in top-k predictions
    relevant_count = sum(1 for label in predicted_labels if label == query_label)
    
    # Total relevant items (excluding query itself)
    total_relevant = sum(1 for label in labels_df["label"] if label == query_label) - 1
    
    if total_relevant == 0:
        return 0.0
    
    return relevant_count / total_relevant


def compute_precision_at_k(
    predicted_indices: List[List[int]],
    k: int,
    labels_df: pd.DataFrame,
    query_idx: int,
) -> float:
    """Compute Precision@k for a single query.
    
    Args:
        predicted_indices: List of predicted indices for each k
        k: Number of top predictions to consider
        labels_df: DataFrame with labels for all images
        query_idx: Index of the query image
        
    Returns:
        Precision@k score
    """
    if len(predicted_indices) < k:
        k = len(predicted_indices)
    
    # Get labels for predicted indices
    predicted_labels = []
    for idx in predicted_indices[:k]:
        if idx < len(labels_df):
            predicted_labels.append(labels_df.iloc[idx]["label"])
        else:
            predicted_labels.append("unknown")
    
    # Get true label for query
    query_label = labels_df.iloc[query_idx]["label"]
    
    # Count relevant items in top-k predictions
    relevant_count = sum(1 for label in predicted_labels if label == query_label)
    
    return relevant_count / k


def compute_average_precision(
    predicted_indices: List[List[int]],
    labels_df: pd.DataFrame,
    query_idx: int,
    max_k: int = 10,
) -> float:
    """Compute Average Precision for a single query.
    
    Args:
        predicted_indices: List of predicted indices
        labels_df: DataFrame with labels for all images
        query_idx: Index of the query image
        max_k: Maximum number of predictions to consider
        
    Returns:
        Average Precision score
    """
    if len(predicted_indices) == 0:
        return 0.0
    
    # Limit to max_k predictions
    predicted_indices = predicted_indices[:max_k]
    
    # Get true label for query
    query_label = labels_df.iloc[query_idx]["label"]
    
    # Create binary relevance vector
    relevance_scores = []
    for idx in predicted_indices:
        if idx < len(labels_df):
            pred_label = labels_df.iloc[idx]["label"]
            relevance_scores.append(1 if pred_label == query_label else 0)
        else:
            relevance_scores.append(0)
    
    if sum(relevance_scores) == 0:
        return 0.0
    
    # Compute Average Precision
    precisions = []
    relevant_count = 0
    
    for i, is_relevant in enumerate(relevance_scores):
        if is_relevant:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precisions.append(precision)
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(precisions)


def evaluate_retrieval(
    predictions: List[List[Tuple[int, float]]],
    labels_df: pd.DataFrame,
    k_values: List[int] = EVAL_K_VALUES,
) -> Dict[str, float]:
    """Evaluate retrieval performance using multiple metrics.
    
    Args:
        predictions: List of predictions for each query
        labels_df: DataFrame with labels for all images
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(predictions) == 0:
        return {}
    
    metrics = {}
    
    # Compute metrics for each k value
    for k in k_values:
        recall_scores = []
        precision_scores = []
        
        for query_idx, pred_list in enumerate(predictions):
            if len(pred_list) == 0:
                continue
            
            # Extract indices from predictions
            pred_indices = [idx for idx, _ in pred_list]
            
            # Compute Recall@k
            recall = compute_recall_at_k(
                [], pred_indices, k, labels_df, query_idx
            )
            recall_scores.append(recall)
            
            # Compute Precision@k
            precision = compute_precision_at_k(
                pred_indices, k, labels_df, query_idx
            )
            precision_scores.append(precision)
        
        # Average across all queries
        if recall_scores:
            metrics[f"Recall@{k}"] = np.mean(recall_scores)
        if precision_scores:
            metrics[f"Precision@{k}"] = np.mean(precision_scores)
    
    # Compute mAP (mean Average Precision)
    ap_scores = []
    for query_idx, pred_list in enumerate(predictions):
        if len(pred_list) == 0:
            continue
        
        pred_indices = [idx for idx, _ in pred_list]
        ap = compute_average_precision(
            pred_indices, labels_df, query_idx, max_k=max(k_values)
        )
        ap_scores.append(ap)
    
    if ap_scores:
        metrics["mAP"] = np.mean(ap_scores)
        for k in k_values:
            # Compute mAP@k
            ap_k_scores = []
            for query_idx, pred_list in enumerate(predictions):
                if len(pred_list) == 0:
                    continue
                
                pred_indices = [idx for idx, _ in pred_list]
                ap_k = compute_average_precision(
                    pred_indices, labels_df, query_idx, max_k=k
                )
                ap_k_scores.append(ap_k)
            
            if ap_k_scores:
                metrics[f"mAP@{k}"] = np.mean(ap_k_scores)
    
    return metrics


def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """Print evaluation results in a formatted table.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    if not metrics:
        print("No evaluation metrics available")
        return
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, score in metrics.items():
        print(f"{metric_name:<15}: {score:.4f}")
    
    print("="*50)
