#!/usr/bin/env python3
"""
Enhanced RAG System with LLM Integration
Uses the optimized vector database and integrates with OpenAI/Claude for better answers
"""

import json
import os
from typing import List, Dict, Any, Optional
from embedding_system import RAGEmbeddingSystem
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedRAGSystem:
    """
    Enhanced RAG system with LLM integration for better answer generation
    """
    
    def __init__(self, vector_db_dir: str = "vector_db", llm_provider: str = "openrouter"):
        """
        Initialize the enhanced RAG system
        
        Args:
            vector_db_dir: Directory containing the vector database
            llm_provider: LLM provider to use ("openrouter", "openai", "claude", or "local")
        """
        self.vector_db_dir = vector_db_dir
        self.llm_provider = llm_provider
        self.embedding_system = None
        self.is_loaded = False
        self.openai_client = None
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.5
        self.low_confidence_threshold = 0.3
        
        # LLM configuration
        self.llm_config = {
            "openrouter": {
                "model": "openai/gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 500,
                "base_url": "https://openrouter.ai/api/v1"
            },
            "openai": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 500
            },
            "claude": {
                "model": "claude-3-haiku-20240307",
                "temperature": 0.3,
                "max_tokens": 500
            }
        }
        
        print(f"Initializing Enhanced RAG System with {llm_provider}...")
    
    def load_system(self):
        """Load the vector database and initialize LLM"""
        if self.is_loaded:
            return
        
        print("Loading vector database...")
        self.embedding_system = RAGEmbeddingSystem()
        self.embedding_system.load_embeddings(self.vector_db_dir)
        print("Vector database loaded successfully!")
        
        # Initialize LLM
        if self.llm_provider == "openrouter":
            self._setup_openrouter()
        elif self.llm_provider == "openai":
            self._setup_openai()
        elif self.llm_provider == "claude":
            self._setup_claude()
        elif self.llm_provider == "local":
            print("Local LLM mode - using template-based responses")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        self.is_loaded = True
        print("Enhanced RAG System loaded successfully!")
    
    def _setup_openrouter(self):
        """Setup OpenRouter API"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Warning: OPENROUTER_API_KEY not found. Using template-based responses.")
            self.llm_provider = "local"
            return
        
        # Configure OpenAI client for OpenRouter
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        print("OpenRouter API configured successfully!")
    
    def _setup_openai(self):
        """Setup OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Using template-based responses.")
            self.llm_provider = "local"
            return

        self.openai_client = OpenAI(api_key=api_key)
        print("OpenAI API configured successfully!")
    
    def _setup_claude(self):
        """Setup Claude API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not found. Using template-based responses.")
            self.llm_provider = "local"
            return
        
        # Claude setup would go here
        print("Claude API configured successfully!")
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using the vector database"""
        if not self.is_loaded:
            self.load_system()
        
        results = self.embedding_system.search_similar(query, top_k)
        return results
    
    def generate_llm_answer(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM"""
        if self.llm_provider == "local":
            return self._generate_template_answer(question, relevant_chunks)
        
        # Prepare context from relevant chunks
        context = self._prepare_context(relevant_chunks)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        try:
            if self.llm_provider == "openrouter":
                return self._call_openrouter(prompt)
            elif self.llm_provider == "openai":
                return self._call_openai(prompt)
            elif self.llm_provider == "claude":
                return self._call_claude(prompt)
        except Exception as e:
            print(f"LLM API error: {e}")
            print("Falling back to template-based response...")
            return self._generate_template_answer(question, relevant_chunks)
    
    def _prepare_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from relevant chunks"""
        context_parts = []
        
        for i, chunk_data in enumerate(relevant_chunks, 1):
            chunk = chunk_data['chunk']
            score = chunk_data['score']
            
            # Clean the text
            text = chunk['text']
            text = text.replace('[Page ', 'Page ').replace(']', '')
            
            context_parts.append(f"Source {i} (Relevance: {score:.2f}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for LLM"""
        prompt = f"""You are a helpful chemistry tutor for 9th grade students. Answer the question based on the provided context from a chemistry textbook.

Question: {question}

Context from Chemistry Textbook:
{context}

Instructions:
1. Answer the question clearly and accurately based on the provided context
2. Use simple language appropriate for 9th grade students
3. If the context doesn't contain enough information, say so
4. Include relevant examples when possible
5. Keep your answer concise but complete
6. Do not make up information not present in the context

Answer:"""
        
        return prompt
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API"""
        config = self.llm_config["openrouter"]
        
        response = self.openai_client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a helpful chemistry tutor for 9th grade students."},
                {"role": "user", "content": prompt}
            ],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        return response.choices[0].message.content.strip()
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        config = self.llm_config["openai"]

        response = self.openai_client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a helpful chemistry tutor for 9th grade students."},
                {"role": "user", "content": prompt}
            ],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )

        return response.choices[0].message.content.strip()
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API"""
        # Claude API implementation would go here
        # For now, return a placeholder
        return "Claude API integration not implemented yet. Using template response."
    
    def _generate_template_answer(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate template-based answer when LLM is not available"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your question."
        
        # Get the best chunk
        best_chunk = relevant_chunks[0]
        best_text = best_chunk['chunk']['text']
        best_score = best_chunk['score']
        
        # Clean the text
        answer = best_text.replace('[Page ', 'Page ').replace(']', '')
        
        # Add confidence indicator
        if best_score > self.high_confidence_threshold:
            confidence_note = "This answer is based on highly relevant information from the chemistry textbook."
        elif best_score > self.medium_confidence_threshold:
            confidence_note = "This answer is based on moderately relevant information from the chemistry textbook."
        else:
            confidence_note = "This answer is based on somewhat relevant information. You may want to verify this information."
        
        return f"{answer}\n\nNote: {confidence_note}"
    
    def generate_citations(self, relevant_chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate citations from relevant chunks"""
        citations = []
        
        for chunk_data in relevant_chunks:
            chunk = chunk_data['chunk']
            metadata = chunk['metadata']
            
            page = metadata.get('pages', ['Unknown'])[0]
            score = chunk_data['score']
            
            citation = f"Page {page} (Relevance: {score:.2f})"
            citations.append(citation)
        
        return citations
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main query method that returns a complete response
        
        Args:
            question: The user's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing answer, citations, and metadata
        """
        if not self.is_loaded:
            self.load_system()
        
        print(f"Processing query: {question}")
        
        # Step 1: Search for relevant chunks
        relevant_chunks = self.search_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your question.",
                "citations": [],
                "confidence": 0.0,
                "num_chunks_found": 0
            }
        
        # Step 2: Calculate confidence based on best match
        best_score = relevant_chunks[0]['score']
        confidence = best_score * 100  # Convert to percentage
        
        # Step 3: Generate answer
        answer = self.generate_llm_answer(question, relevant_chunks)
        
        # Step 4: Generate citations
        citations = self.generate_citations(relevant_chunks)
        
        # Step 5: Prepare response
        response = {
            "question": question,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "num_chunks_found": len(relevant_chunks),
            "llm_provider": self.llm_provider,
            "relevant_chunks": [
                {
                    "text": chunk['chunk']['text'][:200] + "...",
                    "score": chunk['score'],
                    "page": chunk['chunk']['metadata'].get('pages', ['Unknown'])[0]
                }
                for chunk in relevant_chunks
            ]
        }
        
        return response

def main():
    """Test the enhanced RAG system"""
    print("=== TESTING ENHANCED RAG SYSTEM ===")
    
    # Initialize the system
    rag_system = EnhancedRAGSystem(llm_provider="local")  # Use local for testing
    
    # Test queries
    test_queries = [
        "What is chemistry?",
        "What are the branches of chemistry?",
        "What is physical chemistry?",
        "What are the states of matter?",
        "What is a chemical reaction?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Question: {query}")
        
        response = rag_system.query(query)
        
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.1f}%")
        print(f"Citations: {', '.join(response['citations'])}")
        print(f"Chunks found: {response['num_chunks_found']}")
        print("-" * 60)
    
    print("\nEnhanced RAG System test completed!")

if __name__ == "__main__":
    main()
