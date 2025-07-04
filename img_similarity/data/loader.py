"""Dataset loading utilities for image similarity search."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import DEFAULT_ID_COL, DEFAULT_IMG_COL, DEFAULT_LABEL_COL


def load_dataset(
    path: Path | str,
    img_col: str = DEFAULT_IMG_COL,
    label_col: Optional[str] = DEFAULT_LABEL_COL,
    id_col: str = DEFAULT_ID_COL,
) -> pd.DataFrame:
    """Load dataset from CSV file or directory structure.
    
    Supports Stanford-Online-Products CSV format or generic folder-crawl.
    
    Args:
        path: Path to CSV file or directory containing images
        img_col: Column name for image paths
        label_col: Column name for labels (optional)
        id_col: Column name for image IDs
        
    Returns:
        DataFrame with columns [id, image_path, label?]
        
    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If CSV format is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if path.is_file() and path.suffix.lower() == ".csv":
        return _load_csv_dataset(path, img_col, label_col, id_col)
    elif path.is_dir():
        return _load_directory_dataset(path, label_col, id_col)
    else:
        raise ValueError(f"Unsupported path type: {path}")


def _load_csv_dataset(
    csv_path: Path,
    img_col: str,
    label_col: Optional[str],
    id_col: str,
) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if img_col not in df.columns:
        raise ValueError(f"Image column '{img_col}' not found in CSV")
    
    # Create standardized column names
    result_df = pd.DataFrame()
    
    # Add ID column
    if id_col in df.columns:
        result_df[id_col] = df[id_col]
    else:
        result_df[id_col] = range(len(df))
    
    # Add image path column
    result_df[img_col] = df[img_col]
    
    # Add label column if available
    if label_col and label_col in df.columns:
        result_df[label_col] = df[label_col]
    
    return result_df


def _load_directory_dataset(
    dir_path: Path,
    label_col: Optional[str],
    id_col: str,
) -> pd.DataFrame:
    """Load dataset from directory structure."""
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(dir_path.glob(f"**/*{ext}"))
        image_files.extend(dir_path.glob(f"**/*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {dir_path}")
    
    # Create DataFrame
    data = {
        id_col: range(len(image_files)),
        DEFAULT_IMG_COL: [str(f) for f in image_files],
    }
    
    # Add labels based on directory structure if requested
    if label_col:
        labels = []
        for f in image_files:
            # Use parent directory name as label
            label = f.parent.name if f.parent != dir_path else "unknown"
            labels.append(label)
        data[label_col] = labels
    
    return pd.DataFrame(data)
