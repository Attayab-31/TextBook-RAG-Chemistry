#!/usr/bin/env python3
"""
Embedding System for RAG
Converts text chunks into vector embeddings for similarity search
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import os
import pickle
from tqdm import tqdm

class RAGEmbeddingSystem:
    """
    Embedding system for RAG using sentence transformers and FAISS
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_dim: int = 384):
        """
        Initialize the embedding system
        
        Args:
            model_name: Sentence transformer model name
            vector_dim: Dimension of the embedding vectors
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        print(f"Initializing embedding system with model: {model_name}")
        print(f"Vector dimension: {vector_dim}")
    
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_chunks(self, chunks_file: str):
        """Load chunks from JSON file"""
        print(f"Loading chunks from {chunks_file}...")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"Loaded {len(self.chunks)} chunks")
        return self.chunks
    
    def create_embeddings(self, batch_size: int = 32):
        """Create embeddings for all chunks"""
        if not self.model:
            self.load_model()
        
        if not self.chunks:
            raise ValueError("No chunks loaded. Please load chunks first.")
        
        print("Creating embeddings for chunks...")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Create embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(embeddings)
        
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def build_faiss_index(self, index_type: str = "flat"):
        """Build FAISS index for similarity search"""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Please create embeddings first.")
        
        print(f"Building FAISS index (type: {index_type})...")
        
        # Create FAISS index
        if index_type == "flat":
            # Flat index - exact search, slower but more accurate
            self.index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine similarity
        elif index_type == "ivf":
            # IVF index - approximate search, faster but less accurate
            quantizer = faiss.IndexFlatIP(self.vector_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        # Train index if needed (for IVF)
        if index_type == "ivf":
            self.index.train(self.embeddings.astype('float32'))
            self.index.add(self.embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
        return self.index
    
    def save_embeddings(self, output_dir: str = "vector_db"):
        """Save embeddings and index to disk"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Saving embeddings to {output_dir}...")
        
        # Save embeddings
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_file, self.embeddings)
        
        # Save FAISS index
        index_file = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_file)
        
        # Save chunks metadata
        chunks_file = os.path.join(output_dir, "chunks_metadata.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        # Save model info
        model_info = {
            "model_name": self.model_name,
            "vector_dim": self.vector_dim,
            "num_chunks": len(self.chunks),
            "embedding_shape": self.embeddings.shape,
            "index_type": "flat"
        }
        
        info_file = os.path.join(output_dir, "model_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"Embeddings saved successfully to {output_dir}")
        return output_dir
    
    def load_embeddings(self, input_dir: str = "vector_db"):
        """Load embeddings and index from disk"""
        print(f"Loading embeddings from {input_dir}...")
        
        # Load model info
        info_file = os.path.join(input_dir, "model_info.json")
        with open(info_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        self.model_name = model_info["model_name"]
        self.vector_dim = model_info["vector_dim"]
        
        # Load model
        self.load_model()
        
        # Load embeddings
        embeddings_file = os.path.join(input_dir, "embeddings.npy")
        self.embeddings = np.load(embeddings_file)
        
        # Load FAISS index
        index_file = os.path.join(input_dir, "faiss_index.bin")
        self.index = faiss.read_index(index_file)
        
        # Load chunks metadata
        chunks_file = os.path.join(input_dir, "chunks_metadata.json")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"Loaded {len(self.chunks)} chunks with embeddings")
        return True
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using the query"""
        if not self.model or not self.index:
            raise ValueError("Model or index not loaded. Please load embeddings first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar vectors
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                result = {
                    "rank": i + 1,
                    "score": float(score),
                    "chunk": self.chunks[idx],
                    "similarity": float(score)  # Cosine similarity score
                }
                results.append(result)
        
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding system"""
        if not self.embeddings is not None:
            return {}
        
        stats = {
            "model_name": self.model_name,
            "vector_dim": self.vector_dim,
            "num_chunks": len(self.chunks),
            "embedding_shape": self.embeddings.shape,
            "index_size": self.index.ntotal if self.index else 0,
            "memory_usage_mb": self.embeddings.nbytes / (1024 * 1024)
        }
        
        return stats

def main():
    """Main function to build embeddings"""
    print("=== BUILDING EMBEDDINGS FOR RAG SYSTEM ===")
    
    # Initialize embedding system
    embedding_system = RAGEmbeddingSystem(
        model_name="all-MiniLM-L6-v2",  # Fast and efficient model
        vector_dim=384
    )
    
    # Load chunks
    chunks = embedding_system.load_chunks("chemistry_9th_chunks.json")
    
    # Create embeddings
    embeddings = embedding_system.create_embeddings(batch_size=32)
    
    # Build FAISS index
    index = embedding_system.build_faiss_index(index_type="flat")
    
    # Save everything
    output_dir = embedding_system.save_embeddings("vector_db")
    
    # Get and display statistics
    stats = embedding_system.get_embedding_stats()
    
    print("\n=== EMBEDDING SYSTEM STATISTICS ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Test search functionality
    print("\n=== TESTING SEARCH FUNCTIONALITY ===")
    test_query = "What is chemistry?"
    results = embedding_system.search_similar(test_query, top_k=3)
    
    print(f"Test query: '{test_query}'")
    for result in results:
        print(f"Rank {result['rank']}: Score {result['score']:.4f}")
        print(f"Text: {result['chunk']['text'][:100]}...")
        print("-" * 50)
    
    print(f"\nEmbeddings built successfully! Vector database saved to: {output_dir}")
    return embedding_system

if __name__ == "__main__":
    main()
