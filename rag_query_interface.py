#!/usr/bin/env python3
"""
Interactive RAG Query Interface
Allows users to ask questions and get answers from the chemistry RAG system
"""

import os
from enhanced_rag_system import EnhancedRAGSystem

def main(llm_provider="openrouter"):
    """Interactive query interface"""
    print("=== CHEMISTRY RAG QUERY INTERFACE ===")
    print("Ask questions about chemistry from the 9th grade textbook!")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = EnhancedRAGSystem(llm_provider=llm_provider)
    print("RAG system ready!")
    print()
    
    while True:
        try:
            # Get user input
            question = input("Ask a chemistry question: ").strip()
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'help':
                print_help()
                continue
            
            if question.lower() == 'stats':
                print_stats(rag_system)
                continue
            
            # Process the question
            print("\nProcessing your question...")
            response = rag_system.query(question)
            
            # Display the response
            print_response(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

def print_help():
    """Print help information"""
    print("\n=== HELP ===")
    print("Commands:")
    print("  help  - Show this help message")
    print("  stats - Show system statistics")
    print("  quit  - Exit the program")
    print()
    print("Example questions:")
    print("  - What is chemistry?")
    print("  - What are the branches of chemistry?")
    print("  - What is physical chemistry?")
    print("  - What are the states of matter?")
    print("  - What is a chemical reaction?")
    print("  - What is the difference between elements and compounds?")
    print()

def print_stats(rag_system):
    """Print system statistics"""
    print("\n=== SYSTEM STATISTICS ===")
    if rag_system.embedding_system:
        stats = rag_system.embedding_system.get_embedding_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    print()

def print_response(response):
    """Print the response in a formatted way"""
    print("\n" + "="*60)
    print(f"QUESTION: {response['question']}")
    print("="*60)
    
    print(f"\nANSWER:")
    print(response['answer'])
    
    print(f"\nCONFIDENCE: {response['confidence']:.1f}%")
    
    if response['citations']:
        print(f"\nSOURCES:")
        for i, citation in enumerate(response['citations'], 1):
            print(f"  {i}. {citation}")
    
    print(f"\nCHUNKS FOUND: {response['num_chunks_found']}")
    print(f"LLM PROVIDER: {response['llm_provider']}")
    
    print("\n" + "="*60)
    print()

def setup_openrouter():
    """Setup OpenRouter API key"""
    print("=== OPENROUTER API SETUP ===")
    print("To use OpenRouter for better answers, you need an API key.")
    print("1. Get an API key from: https://openrouter.ai/")
    print("2. Set it as an environment variable:")
    print("   Windows: set OPENROUTER_API_KEY=your_key_here")
    print("   Linux/Mac: export OPENROUTER_API_KEY=your_key_here")
    print("3. Or create a .env file with: OPENROUTER_API_KEY=your_key_here")
    print()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print("SUCCESS: OpenRouter API key found!")
        return True
    else:
        print("WARNING: OpenRouter API key not found. Using template-based responses.")
        return False

if __name__ == "__main__":
    # Check for OpenRouter API key
    has_openrouter = setup_openrouter()
    
    if has_openrouter:
        choice = input("Use OpenRouter for better answers? (y/n): ").strip().lower()
        llm_provider = "openrouter" if choice == 'y' else "local"
    else:
        llm_provider = "local"

    main(llm_provider=llm_provider)
