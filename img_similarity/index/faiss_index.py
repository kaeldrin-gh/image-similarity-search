"""FAISS-based approximate nearest neighbor search."""

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from ..config import EMBEDDING_DIM


class FaissIndexer:
    """FAISS-based indexer for approximate nearest neighbor search."""
    
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        """Initialize the FAISS indexer.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index: faiss.Index = None
        self.is_trained = False
    
    def build(self, vectors: np.ndarray) -> None:
        """Build the FAISS index from embedding vectors.
        
        Args:
            vectors: Array of embedding vectors with shape (n_vectors, embedding_dim)
            
        Raises:
            ValueError: If vectors have wrong shape
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected vectors with shape (n, {self.embedding_dim}), "
                f"got {vectors.shape}"
            )
        
        # Ensure vectors are float32 (required by FAISS)
        vectors = vectors.astype(np.float32)
        
        # Create FAISS index
        # Using L2 distance for normalized vectors (equivalent to cosine similarity)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add vectors to index
        self.index.add(vectors)
        self.is_trained = True
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def query(self, vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Query the index for k nearest neighbors.
        
        Args:
            vector: Query vector with shape (embedding_dim,)
            k: Number of nearest neighbors to return
            
        Returns:
            List of (index, distance) tuples for the k nearest neighbors
            
        Raises:
            RuntimeError: If index is not built
            ValueError: If vector has wrong shape
        """
        if not self.is_trained or self.index is None:
            raise RuntimeError("Index must be built before querying")
        
        if vector.ndim != 1 or vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected vector with shape ({self.embedding_dim},), "
                f"got {vector.shape}"
            )
        
        # Ensure vector is float32 and 2D for FAISS
        vector = vector.astype(np.float32).reshape(1, -1)
        
        # Limit k to available vectors
        k = min(k, self.index.ntotal)
        
        # Search for k nearest neighbors
        distances, indices = self.index.search(vector, k)
        
        # Convert to list of tuples
        results = []
        for i in range(k):
            results.append((int(indices[0][i]), float(distances[0][i])))
        
        return results
    
    def save(self, path: Path | str) -> None:
        """Save the index to disk.
        
        Args:
            path: Path to save the index file
            
        Raises:
            RuntimeError: If index is not built
        """
        if not self.is_trained or self.index is None:
            raise RuntimeError("Index must be built before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path))
        print(f"Saved FAISS index to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> "FaissIndexer":
        """Load an index from disk.
        
        Args:
            path: Path to the index file
            
        Returns:
            Loaded FaissIndexer instance
            
        Raises:
            FileNotFoundError: If index file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        # Load the index
        index = faiss.read_index(str(path))
        
        # Create indexer instance
        indexer = cls(embedding_dim=index.d)
        indexer.index = index
        indexer.is_trained = True
        
        print(f"Loaded FAISS index from {path} with {index.ntotal} vectors")
        return indexer
    
    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
