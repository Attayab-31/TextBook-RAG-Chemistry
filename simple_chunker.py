"""
Simplified Text Chunking System for PCTB Chemistry Book
No chapter detection - just clean, simple chunking
"""
import json
import re
from typing import List, Dict, Any

class SimplePCTBChunker:
    """
    Simple chunker without chapter detection - just clean chunking
    """
    
    def __init__(self, chunk_size: int = 750, overlap: int = 75):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text_into_chunks(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks without chapter detection
        """
        # Remove page reference for chunking
        clean_text = re.sub(r'^\[Page \d+\]\s*', '', text)
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': page_num,
                    'chunk_index': chunk_index,
                    'start_char': 0,
                    'end_char': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page': page_num,
                'chunk_index': chunk_index,
                'start_char': 0,
                'end_char': len(current_chunk)
            })
        
        return chunks
    
    def process_page(self, page_data: Dict, class_level: str = "9th", 
                    stream: str = "Science", subject: str = "Chemistry") -> List[Dict[str, Any]]:
        """
        Process a single page into chunks with simple metadata
        """
        page_num = page_data['page']
        text = page_data['text']
        
        # Split into chunks
        chunks = self.split_text_into_chunks(text, page_num)
        
        # Add simple metadata to each chunk
        processed_chunks = []
        for chunk_data in chunks:
            chunk_id = f"{subject.lower()}_{class_level}_chunk_{chunk_data['chunk_index']}"
            
            # Create final chunk structure
            final_chunk = {
                'chunk_id': chunk_id,
                'text': f"[Page {page_num}] {chunk_data['text']}",
                'metadata': {
                    'class': class_level,
                    'stream': stream,
                    'subject': subject,
                    'pages': [page_num],
                    'chunk_index': chunk_data['chunk_index'],
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk_data['text']),
                    'overlap': self.overlap
                },
                'original_page': page_num,
                'chunk_index': chunk_data['chunk_index']
            }
            
            processed_chunks.append(final_chunk)
        
        return processed_chunks
    
    def process_book(self, json_file_path: str, output_file_path: str = None) -> List[Dict[str, Any]]:
        """
        Process entire book into simple chunks
        """
        print("=== PROCESSING CHEMISTRY BOOK INTO SIMPLE CHUNKS ===")
        
        # Load the cleaned text data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
        
        print(f"Loaded {len(pages_data)} pages from {json_file_path}")
        
        all_chunks = []
        total_pages = len(pages_data)
        
        for i, page_data in enumerate(pages_data, 1):
            if i % 20 == 0 or i == total_pages:
                print(f"Processing page {i}/{total_pages} (Book page {page_data['page']})...", end="\r")
            
            # Process this page into chunks
            page_chunks = self.process_page(page_data)
            all_chunks.extend(page_chunks)
        
        print(f"\nCompleted processing {total_pages} pages!")
        print(f"Created {len(all_chunks)} total chunks")
        
        # Save to output file
        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            print(f"Simple chunks saved to {output_file_path}")
        
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the simple chunking process
        """
        total_chunks = len(chunks)
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_chunk_size = total_chars // total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'average_chunk_size': avg_chunk_size,
            'chunk_size_range': {
                'min': min(len(chunk['text']) for chunk in chunks),
                'max': max(len(chunk['text']) for chunk in chunks)
            }
        }

def main():
    """
    Main function to run the simple chunking process
    """
    # Initialize simple chunker
    chunker = SimplePCTBChunker(chunk_size=750, overlap=75)
    
    # Process the Chemistry book (using enterprise cleaned data)
    chunks = chunker.process_book(
        json_file_path='chemistry_9th.json',
        output_file_path='chemistry_9th_chunks.json'
    )
    
    # Get and display statistics
    stats = chunker.get_chunking_stats(chunks)
    
    print("\n=== SIMPLE CHUNKING STATISTICS ===")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Average chunk size: {stats['average_chunk_size']} characters")
    print(f"Chunk size range: {stats['chunk_size_range']['min']}-{stats['chunk_size_range']['max']} characters")
    
    print("\n=== SAMPLE SIMPLE CHUNKS ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['metadata']['pages']}")
        print(f"Size: {len(chunk['text'])} characters")
        print(f"Text preview: {chunk['text'][:150]}...")
        print(f"Metadata: {chunk['metadata']}")

if __name__ == "__main__":
    main()
