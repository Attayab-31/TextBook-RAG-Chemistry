"""
Enterprise-Grade Text Cleaner for OCR Output
Industry Standard Multi-Stage Pipeline
"""
import re
import json
import ftfy
import unidecode
import regex
from typing import List, Tuple, Dict, Any, Optional
from textblob import TextBlob
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
try:
    from enhanced_spell_corrector import EnhancedSpellCorrector
except ImportError:
    EnhancedSpellCorrector = None

class EnterpriseTextCleaner:
    """
    Industry-standard text cleaning pipeline for OCR output
    Uses multi-stage approach with quality validation
    """
    
    def __init__(self):
        self.quality_threshold = 10.0  # Very low threshold to keep most pages
        self.min_text_length = 5  # Very low minimum to keep most pages
        
        # Initialize enhanced spell corrector if available
        self.spell_corrector = EnhancedSpellCorrector() if EnhancedSpellCorrector else None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def stage1_encoding_fixes(self, text: str) -> str:
        """
        Stage 1: Fix encoding issues and mojibake
        """
        if not text:
            return text
        
        # Fix encoding issues (mojibake, etc.)
        text = ftfy.fix_text(text)
        
        # Normalize unicode characters
        text = unidecode.unidecode(text)
        
        return text
    
    def stage2_ocr_artifact_removal(self, text: str) -> str:
        """
        Stage 2: Remove OCR-specific artifacts
        """
        if not text:
            return text
        
        # Remove page references and headers
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'PCTB|Punjab Curriculum|Textbook Board|Government of Punjab', '', text, flags=re.IGNORECASE)
        
        # Remove specific OCR artifacts (exact matches)
        specific_artifacts = [
            r'Chapter D api One a ; a, un " 4 Fy Yo See SPhaselChanges 4 e',
            r'es es ade fe c',
            r'om y- , ale \) l 00 atte',
            r'ter material things',
            r'ation with state symbols',
            r's crystalline forms',
            r'Graphite isa good conductor',
            r'I\) Chapter 2\) \) i m PU Cal ata oliection and: Analysis',
            r'4\. the vast and complex subject',
            r'3 Matter exists mainly',
            r'5 Matter also exists',
            r'6\.',
        ]
        
        for pattern in specific_artifacts:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Remove chemistry-specific repetitive content
        chemistry_artifacts = [
            r'Student learning Outcomes',
            r'After studying this chapter',
            r'students will be able to',
            r'Some examples may include',
            r'In the field of chemistry',
            r'This standardized system ensures',
            r'facilitating communication and collaboration',
        ]
        
        for pattern in chemistry_artifacts:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        return text
    
    def stage3_unicode_normalization(self, text: str) -> str:
        """
        Stage 3: Advanced unicode and character normalization
        """
        if not text:
            return text
        
        # Remove control characters and symbols
        text = regex.sub(r'\p{C}', '', text)  # Remove control characters
        text = regex.sub(r'\p{So}', '', text)  # Remove symbols
        
        # Fix common OCR character substitutions
        ocr_fixes = {
            '0': 'O',  # Zero to O in text context
            '1': 'I',  # One to I in text context  
            '5': 'S',  # Five to S in text context
            '8': 'B',  # Eight to B in text context
            '6': 'G',  # Six to G in text context
            '9': 'g',  # Nine to g in text context
            '|': 'I',  # Pipe to I
            '[]': 'o',  # Empty brackets to o
            '()': 'o',  # Empty parentheses to o
        }
        
        for wrong, correct in ocr_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def stage4_nlp_cleaning(self, text: str) -> str:
        """
        Stage 4: NLP-based cleaning and correction
        """
        if not text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove lines that are too short or contain mostly artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Keep lines that are substantial
            if (len(line) > 10 and 
                not re.search(r'^[0-9\s]+$', line) and  # Not just numbers
                not re.search(r'^[a-zA-Z\s]{1,2}$', line) and  # Not just 1-2 letters
                not re.search(r'^[a-zA-Z]{1,2}\s+[a-zA-Z]{1,2}\s+[a-zA-Z]{1,2}$', line)):  # Not just short fragments
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove any remaining single characters at beginning/end
        text = re.sub(r'^\s*[a-zA-Z]\s+', '', text)
        text = re.sub(r'\s+[a-zA-Z]\s*$', '', text)
        
        return text
    
    def stage5_spell_correction(self, text: str) -> str:
        """
        Stage 5: Enhanced spell correction using chemistry-specific corrector
        """
        if not text or len(text) < 10 or not self.spell_corrector:
            return text
        
        try:
            # Use enhanced spell corrector for chemistry terms
            corrected = self.spell_corrector.correct_text(text)
            return corrected
        except Exception:
            # If correction fails, return original text
            return text
    
    def stage6_quality_validation(self, text: str) -> Tuple[str, bool]:
        """
        Stage 6: Quality validation and filtering
        """
        if not text or len(text) < self.min_text_length:
            return text, False
        
        try:
            # Calculate readability score
            readability_score = flesch_reading_ease(text)
            
            # Check if text meets quality threshold
            if readability_score < self.quality_threshold:
                return text, False
            
            return text, True
        except Exception:
            # If quality check fails, accept the text
            return text, True
    
    def clean_text(self, text: str) -> str:
        """
        Main cleaning pipeline - industry standard approach
        """
        if not text:
            return text
        
        original_length = len(text)
        
        # Stage 1: Encoding fixes
        text = self.stage1_encoding_fixes(text)
        
        # Stage 2: OCR artifact removal
        text = self.stage2_ocr_artifact_removal(text)
        
        # Stage 3: Unicode normalization
        text = self.stage3_unicode_normalization(text)
        
        # Stage 4: NLP cleaning
        text = self.stage4_nlp_cleaning(text)
        
        # Stage 5: Spell correction
        text = self.stage5_spell_correction(text)
        
        # Stage 6: Quality validation
        text, is_quality_good = self.stage6_quality_validation(text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def clean_page(self, page_num: int, text: str) -> Tuple[int, str, Dict[str, Any]]:
        """
        Clean a single page and return detailed statistics
        """
        original_length = len(text)
        original_lines = len(text.split('\n'))
        
        cleaned_text = self.clean_text(text)
        
        final_length = len(cleaned_text)
        final_lines = len(cleaned_text.split('\n'))
        
        # Calculate quality metrics
        try:
            readability_score = flesch_reading_ease(cleaned_text)
            grade_level = flesch_kincaid_grade(cleaned_text)
        except:
            readability_score = 0
            grade_level = 0
        
        stats = {
            "original_length": original_length,
            "final_length": final_length,
            "original_lines": original_lines,
            "final_lines": final_lines,
            "compression_ratio": final_length / original_length if original_length > 0 else 0,
            "readability_score": readability_score,
            "grade_level": grade_level,
            "quality_passed": readability_score >= self.quality_threshold
        }
        
        return page_num, cleaned_text, stats
    
    def parse_text_file(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Parse the text file format from extract.py
        """
        pages_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by page separators
        page_sections = content.split('=== BOOK PAGE ')
        
        for section in page_sections[1:]:  # Skip first empty section
            if '===' in section:
                # Extract page number and text
                lines = section.split('\n')
                page_line = lines[0]
                page_num = int(page_line.split(' ===')[0])
                
                # Extract text (everything after the page header)
                text_lines = []
                in_text = False
                for line in lines[1:]:
                    if line.strip() == '=' * 50:
                        break
                    if in_text or line.strip():
                        text_lines.append(line)
                        in_text = True
                
                text = '\n'.join(text_lines).strip()
                
                if text:  # Only add if there's actual text
                    pages_data.append({
                        'page': page_num,
                        'text': text
                    })
        
        return pages_data
    
    def process_all_pages(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Process all pages with enterprise-grade cleaning
        """
        print("=== ENTERPRISE TEXT CLEANING PIPELINE ===")
        print("Using industry-standard multi-stage approach...")
        
        # Load the extracted text data (handle both JSON and text formats)
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                pages_data = json.load(f)
        else:
            # Handle text format from extract.py
            pages_data = self.parse_text_file(input_file)
        
        print(f"Loaded {len(pages_data)} pages from {input_file}")
        
        cleaned_pages = []
        total_stats = {
            "total_pages": len(pages_data),
            "processed_pages": 0,
            "quality_passed": 0,
            "total_original_length": 0,
            "total_final_length": 0,
            "average_readability": 0,
            "average_grade_level": 0
        }
        
        readability_scores = []
        grade_levels = []
        
        for i, page_data in enumerate(pages_data, 1):
            if i % 20 == 0 or i == len(pages_data):
                print(f"Processing page {i}/{len(pages_data)} (Book page {page_data['page']})...", end="\r")
            
            page_num = page_data['page']
            text = page_data['text']
            
            # Clean the page
            cleaned_page_num, cleaned_text, stats = self.clean_page(page_num, text)
            
            # Update statistics
            total_stats["processed_pages"] += 1
            total_stats["total_original_length"] += stats["original_length"]
            total_stats["total_final_length"] += stats["final_length"]
            
            if stats["quality_passed"]:
                total_stats["quality_passed"] += 1
                readability_scores.append(stats["readability_score"])
                grade_levels.append(stats["grade_level"])
            
            # Keep ALL pages regardless of quality (but still clean them)
            if len(cleaned_text) > 0:  # Only skip completely empty pages
                cleaned_pages.append({
                    "page": cleaned_page_num,
                    "text": cleaned_text,
                    "stats": stats
                })
        
        # Calculate final statistics
        if readability_scores:
            total_stats["average_readability"] = sum(readability_scores) / len(readability_scores)
            total_stats["average_grade_level"] = sum(grade_levels) / len(grade_levels)
        
        total_stats["compression_ratio"] = (
            total_stats["total_final_length"] / total_stats["total_original_length"] 
            if total_stats["total_original_length"] > 0 else 0
        )
        
        print(f"\nEnterprise cleaning completed!")
        print(f"Processed: {total_stats['processed_pages']} pages")
        print(f"Quality passed: {total_stats['quality_passed']} pages")
        print(f"Average readability: {total_stats['average_readability']:.1f}")
        print(f"Average grade level: {total_stats['average_grade_level']:.1f}")
        print(f"Compression ratio: {total_stats['compression_ratio']:.2f}")
        
        # Save cleaned data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)
        
        print(f"Enterprise-cleaned text saved to {output_file}")
        
        return total_stats

def main():
    """
    Main function to run enterprise text cleaning
    """
    cleaner = EnterpriseTextCleaner()
    
    # Process the Chemistry book with enterprise-grade cleaning
    stats = cleaner.process_all_pages(
        input_file='sample_extracted_text.txt',
        output_file='chemistry_9th.json'
    )
    
    print("\n=== ENTERPRISE CLEANING STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
