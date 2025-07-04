"""Test FAISS indexing functionality."""

import numpy as np
import pytest
from pathlib import Path

from img_similarity.index.faiss_index import FaissIndexer


class TestFaissIndexer:
    """Test cases for FAISS indexing functionality."""
    
    def test_build_index(self, sample_embeddings):
        """Test building FAISS index."""
        indexer = FaissIndexer(embedding_dim=2048)
        indexer.build(sample_embeddings)
        
        assert indexer.is_trained
        assert indexer.size == len(sample_embeddings)
    
    def test_build_invalid_shape(self):
        """Test building index with invalid embedding shape."""
        indexer = FaissIndexer(embedding_dim=2048)
        invalid_embeddings = np.random.rand(10, 1024)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected vectors with shape"):
            indexer.build(invalid_embeddings)
    
    def test_query_index(self, sample_embeddings):
        """Test querying FAISS index."""
        indexer = FaissIndexer(embedding_dim=2048)
        indexer.build(sample_embeddings)
        
        # Query with first embedding
        query_vector = sample_embeddings[0]
        results = indexer.query(query_vector, k=3)
        
        assert len(results) == 3
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(dist, float) for _, dist in results)
        
        # First result should be the query vector itself (distance ~0)
        assert results[0][0] == 0
        assert results[0][1] < 1e-5  # Very small distance
    
    def test_query_without_build(self):
        """Test querying index before building."""
        indexer = FaissIndexer(embedding_dim=2048)
        query_vector = np.random.rand(2048)
        
        with pytest.raises(RuntimeError, match="Index must be built"):
            indexer.query(query_vector)
    
    def test_query_invalid_shape(self, sample_embeddings):
        """Test querying with invalid vector shape."""
        indexer = FaissIndexer(embedding_dim=2048)
        indexer.build(sample_embeddings)
        
        invalid_query = np.random.rand(1024)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected vector with shape"):
            indexer.query(invalid_query)
    
    def test_save_and_load_index(self, sample_embeddings, temp_dir):
        """Test saving and loading FAISS index."""
        # Build and save index
        indexer = FaissIndexer(embedding_dim=2048)
        indexer.build(sample_embeddings)
        
        index_path = temp_dir / "test_index.faiss"
        indexer.save(index_path)
        
        assert index_path.exists()
        
        # Load index
        loaded_indexer = FaissIndexer.load(index_path)
        
        assert loaded_indexer.is_trained
        assert loaded_indexer.size == len(sample_embeddings)
        
        # Test that loaded index works
        query_vector = sample_embeddings[0]
        results = loaded_indexer.query(query_vector, k=3)
        assert len(results) == 3
    
    def test_save_without_build(self, temp_dir):
        """Test saving index before building."""
        indexer = FaissIndexer(embedding_dim=2048)
        index_path = temp_dir / "test_index.faiss"
        
        with pytest.raises(RuntimeError, match="Index must be built"):
            indexer.save(index_path)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent index file."""
        with pytest.raises(FileNotFoundError):
            FaissIndexer.load("/nonexistent/index.faiss")
    
    def test_query_k_larger_than_size(self, sample_embeddings):
        """Test querying with k larger than index size."""
        indexer = FaissIndexer(embedding_dim=2048)
        indexer.build(sample_embeddings)
        
        query_vector = sample_embeddings[0]
        results = indexer.query(query_vector, k=100)  # More than 10 vectors
        
        # Should return all available vectors
        assert len(results) == len(sample_embeddings)
