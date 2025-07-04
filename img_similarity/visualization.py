"""Visualization utilities for image similarity search."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def display_query_results(
    query_image_path: str,
    similar_images: List[Tuple[str, float]],
    max_images: int = 10,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Path] = None,
) -> None:
    """Display query image and its similar images.
    
    Args:
        query_image_path: Path to the query image
        similar_images: List of (image_path, similarity_score) tuples
        max_images: Maximum number of similar images to display
        figsize: Figure size for matplotlib
        save_path: Optional path to save the plot
    """
    # Limit number of images to display
    similar_images = similar_images[:max_images]
    
    # Calculate subplot layout
    n_images = len(similar_images) + 1  # +1 for query image
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_images == 1 else axes
    else:
        axes = axes.flatten()
    
    # Display query image
    try:
        query_img = Image.open(query_image_path)
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error loading\n{query_image_path}", 
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("Query Image (Error)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
    
    # Display similar images
    for i, (img_path, score) in enumerate(similar_images, 1):
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"#{i}\nDistance: {score:.3f}", fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading\n{Path(img_path).name}", 
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"#{i} (Error)", fontsize=10)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def plot_embedding_distribution(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    max_samples: int = 1000,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
) -> None:
    """Plot embedding distribution using t-SNE or PCA.
    
    Args:
        embeddings: Embedding vectors
        labels: Optional labels for coloring
        max_samples: Maximum number of samples to plot
        figsize: Figure size for matplotlib
        save_path: Optional path to save the plot
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Subsample if too many points
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        if labels:
            labels = [labels[i] for i in indices]
    
    # Apply dimensionality reduction
    if embeddings.shape[1] > 50:
        # First apply PCA to reduce to 50 dimensions
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Then apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_pca)
    else:
        # Apply t-SNE directly
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels:
        # Plot with colors based on labels
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = [l == label for l in labels]
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[color], label=label, alpha=0.6, s=10)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Plot without colors
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  alpha=0.6, s=10, c='blue')
    
    ax.set_title("Embedding Distribution (t-SNE)", fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved embedding distribution plot to {save_path}")
    
    plt.show()


def plot_evaluation_metrics(
    metrics: dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> None:
    """Plot evaluation metrics as bar chart.
    
    Args:
        metrics: Dictionary of evaluation metrics
        figsize: Figure size for matplotlib
        save_path: Optional path to save the plot
    """
    if not metrics:
        print("No metrics to plot")
        return
    
    # Separate metrics by type
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith("Recall")}
    precision_metrics = {k: v for k, v in metrics.items() if k.startswith("Precision")}
    map_metrics = {k: v for k, v in metrics.items() if k.startswith("mAP")}
    
    # Create subplots
    n_plots = sum([bool(recall_metrics), bool(precision_metrics), bool(map_metrics)])
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot Recall metrics
    if recall_metrics:
        names = list(recall_metrics.keys())
        values = list(recall_metrics.values())
        axes[plot_idx].bar(names, values, color='skyblue')
        axes[plot_idx].set_title("Recall Metrics")
        axes[plot_idx].set_ylabel("Score")
        axes[plot_idx].set_ylim(0, 1)
        plot_idx += 1
    
    # Plot Precision metrics
    if precision_metrics:
        names = list(precision_metrics.keys())
        values = list(precision_metrics.values())
        axes[plot_idx].bar(names, values, color='lightgreen')
        axes[plot_idx].set_title("Precision Metrics")
        axes[plot_idx].set_ylabel("Score")
        axes[plot_idx].set_ylim(0, 1)
        plot_idx += 1
    
    # Plot mAP metrics
    if map_metrics:
        names = list(map_metrics.keys())
        values = list(map_metrics.values())
        axes[plot_idx].bar(names, values, color='lightcoral')
        axes[plot_idx].set_title("mAP Metrics")
        axes[plot_idx].set_ylabel("Score")
        axes[plot_idx].set_ylim(0, 1)
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved evaluation metrics plot to {save_path}")
    
    plt.show()
