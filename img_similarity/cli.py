"""Command-line interface for image similarity search."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMG_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TOP_K,
    EMBEDDINGS_FILE,
    INDEX_FILE,
)
from .data.loader import load_dataset
from .evaluation import evaluate_retrieval, print_evaluation_results
from .index.faiss_index import FaissIndexer
from .vision.embeddings import EmbeddingExtractor

app = typer.Typer(help="Image Similarity Search CLI")
console = Console()


@app.command()
def extract(
    data_dir: Path = typer.Argument(..., help="Path to data directory or CSV file"),
    out: Path = typer.Option(EMBEDDINGS_FILE, help="Output file for embeddings"),
    img_col: str = typer.Option(DEFAULT_IMG_COL, help="Image column name"),
    label_col: Optional[str] = typer.Option(DEFAULT_LABEL_COL, help="Label column name"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, help="Batch size for processing"),
    model_name: str = typer.Option("resnet50", help="Model name to use"),
) -> None:
    """Extract embeddings from images."""
    console.print(f"[bold blue]Loading dataset from {data_dir}[/bold blue]")
    
    try:
        # Load dataset
        df = load_dataset(data_dir, img_col=img_col, label_col=label_col)
        console.print(f"Loaded {len(df)} images")
        
        # Initialize embedding extractor
        console.print(f"[bold blue]Initializing {model_name} model[/bold blue]")
        extractor = EmbeddingExtractor(
            model_name=model_name,
            batch_size=batch_size,
        )
        
        # Extract embeddings
        console.print("[bold blue]Extracting embeddings[/bold blue]")
        image_paths = df[img_col].tolist()
        embeddings = extractor.extract_embeddings(image_paths)
        
        # Save embeddings and metadata
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, embeddings)
        
        # Save metadata
        metadata_file = out.with_suffix(".metadata.csv")
        df.to_csv(metadata_file, index=False)
        
        console.print(f"[bold green]Saved {len(embeddings)} embeddings to {out}[/bold green]")
        console.print(f"[bold green]Saved metadata to {metadata_file}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def index(
    vec: Path = typer.Argument(..., help="Path to embeddings file"),
    out: Path = typer.Option(INDEX_FILE, help="Output file for index"),
) -> None:
    """Build search index from embeddings."""
    console.print(f"[bold blue]Loading embeddings from {vec}[/bold blue]")
    
    try:
        # Load embeddings
        embeddings = np.load(vec)
        console.print(f"Loaded {len(embeddings)} embeddings")
        
        # Build index
        console.print("[bold blue]Building FAISS index[/bold blue]")
        indexer = FaissIndexer()
        indexer.build(embeddings)
        
        # Save index
        out.parent.mkdir(parents=True, exist_ok=True)
        indexer.save(out)
        
        console.print(f"[bold green]Saved index to {out}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def query(
    image: Path = typer.Argument(..., help="Path to query image"),
    index_file: Path = typer.Argument(..., help="Path to index file"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Number of similar images to return"),
    metadata: Optional[Path] = typer.Option(None, help="Path to metadata CSV file"),
    save_results: Optional[Path] = typer.Option(None, help="Path to save results JSON"),
) -> None:
    """Query for similar images."""
    console.print(f"[bold blue]Loading index from {index_file}[/bold blue]")
    
    try:
        # Load index
        indexer = FaissIndexer.load(index_file)
        
        # Extract query embedding
        console.print(f"[bold blue]Extracting embedding for {image}[/bold blue]")
        extractor = EmbeddingExtractor()
        query_embedding = extractor.extract_single_embedding(str(image))
        
        # Search for similar images
        console.print("[bold blue]Searching for similar images[/bold blue]")
        results = indexer.query(query_embedding, k=top_k)
        
        # Load metadata if available
        metadata_df = None
        if metadata:
            metadata_df = pd.read_csv(metadata)
        
        # Display results
        table = Table(title=f"Top {top_k} Similar Images")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Index", justify="right", style="magenta")
        table.add_column("Distance", justify="right", style="green")
        
        if metadata_df is not None:
            table.add_column("Image Path", style="blue")
            if DEFAULT_LABEL_COL in metadata_df.columns:
                table.add_column("Label", style="yellow")
        
        for rank, (idx, distance) in enumerate(results, 1):
            row = [str(rank), str(idx), f"{distance:.4f}"]
            
            if metadata_df is not None and idx < len(metadata_df):
                row.append(str(metadata_df.iloc[idx][DEFAULT_IMG_COL]))
                if DEFAULT_LABEL_COL in metadata_df.columns:
                    row.append(str(metadata_df.iloc[idx][DEFAULT_LABEL_COL]))
            
            table.add_row(*row)
        
        console.print(table)
        
        # Save results if requested
        if save_results:
            results_data = {
                "query_image": str(image),
                "top_k": top_k,
                "results": [
                    {"rank": i+1, "index": idx, "distance": float(dist)}
                    for i, (idx, dist) in enumerate(results)
                ]
            }
            
            if metadata_df is not None:
                for i, (idx, _) in enumerate(results):
                    if idx < len(metadata_df):
                        results_data["results"][i]["image_path"] = str(
                            metadata_df.iloc[idx][DEFAULT_IMG_COL]
                        )
                        if DEFAULT_LABEL_COL in metadata_df.columns:
                            results_data["results"][i]["label"] = str(
                                metadata_df.iloc[idx][DEFAULT_LABEL_COL]
                            )
            
            with open(save_results, "w") as f:
                json.dump(results_data, f, indent=2)
            
            console.print(f"[bold green]Results saved to {save_results}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    index_file: Path = typer.Argument(..., help="Path to index file"),
    metadata: Path = typer.Argument(..., help="Path to metadata CSV file"),
    embeddings: Path = typer.Argument(..., help="Path to embeddings file"),
    num_queries: int = typer.Option(100, help="Number of queries to evaluate"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Maximum k for evaluation"),
) -> None:
    """Evaluate retrieval performance."""
    console.print("[bold blue]Loading data for evaluation[/bold blue]")
    
    try:
        # Load index and metadata
        indexer = FaissIndexer.load(index_file)
        metadata_df = pd.read_csv(metadata)
        embeddings_array = np.load(embeddings)
        
        # Check if labels are available
        if DEFAULT_LABEL_COL not in metadata_df.columns:
            console.print("[bold yellow]Warning: No labels found. Evaluation not possible.[/bold yellow]")
            console.print("[bold yellow]Consider running qualitative evaluation instead.[/bold yellow]")
            return
        
        # Limit number of queries
        num_queries = min(num_queries, len(metadata_df))
        
        console.print(f"[bold blue]Evaluating with {num_queries} queries[/bold blue]")
        
        # Run evaluation
        predictions = []
        
        for i in track(range(num_queries), description="Running queries..."):
            query_embedding = embeddings_array[i]
            results = indexer.query(query_embedding, k=top_k + 1)  # +1 to exclude self
            
            # Remove self from results
            filtered_results = [(idx, dist) for idx, dist in results if idx != i]
            if len(filtered_results) > top_k:
                filtered_results = filtered_results[:top_k]
            
            predictions.append(filtered_results)
        
        # Compute metrics
        console.print("[bold blue]Computing evaluation metrics[/bold blue]")
        metrics = evaluate_retrieval(predictions, metadata_df)
        
        # Print results
        print_evaluation_results(metrics)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
