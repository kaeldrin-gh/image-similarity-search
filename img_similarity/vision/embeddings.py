"""Deep learning embedding extraction for images."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from ..config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    EMBEDDING_DIM,
    NUMPY_SEED,
    TORCH_SEED,
)


class ImageDataset(Dataset):
    """Dataset for loading images from file paths."""
    
    def __init__(self, image_paths: List[str], transform: Optional[transforms.Compose] = None):
        """Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get image and path by index.
        
        Args:
            idx: Index of image to load
            
        Returns:
            Tuple of (image tensor, image path)
        """
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_path


class EmbeddingExtractor:
    """CNN-based feature extractor for image embeddings."""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        batch_size: int = DEFAULT_BATCH_SIZE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: Optional[str] = None,
    ):
        """Initialize the embedding extractor.
        
        Args:
            model_name: Name of the CNN model to use
            batch_size: Batch size for processing
            image_size: Input image size
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Set random seeds for reproducibility
        torch.manual_seed(TORCH_SEED)
        np.random.seed(NUMPY_SEED)
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def _load_model(self) -> nn.Module:
        """Load and prepare the CNN model.
        
        Returns:
            CNN model with modified final layer
        """
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Extract embeddings for a list of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of L2-normalized embeddings with shape (n_images, embedding_dim)
        """
        dataset = ImageDataset(image_paths, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device.type == "cuda" else False,
        )
        
        embeddings = []
        failed_images = []
        
        with torch.no_grad():
            for batch_images, batch_paths in dataloader:
                try:
                    batch_images = batch_images.to(self.device)
                    batch_embeddings = self.model(batch_images)
                    
                    # Flatten and normalize embeddings
                    batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings, p=2, dim=1
                    )
                    
                    embeddings.append(batch_embeddings.cpu().numpy())
                    
                except Exception as e:
                    print(f"Failed to process batch: {e}")
                    failed_images.extend(batch_paths)
        
        if failed_images:
            print(f"Warning: Failed to process {len(failed_images)} images")
        
        if not embeddings:
            raise RuntimeError("No embeddings were successfully extracted")
        
        return np.vstack(embeddings)
    
    def extract_single_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            L2-normalized embedding vector
        """
        embeddings = self.extract_embeddings([image_path])
        return embeddings[0]
