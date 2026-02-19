import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import cv2
import numpy as np

pdf_path = "./books/class_9/Chemistry_9th.pdf"

# Page offset configuration
# PDF page 5 = Book page 1, so offset is 4
PAGE_OFFSET = 4  # Adjust this based on your book

def setup_tesseract():
    """Setup Tesseract path automatically"""
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
        r"C:\Tesseract-OCR\tesseract.exe"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"Found Tesseract at: {path}")
            return True
    
    print("Tesseract not found in common locations.")
    print("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
    return False


def preprocess_image_for_ocr(img):
    """
    Advanced image preprocessing for better OCR results
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding for better text contrast
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to PIL Image
    processed_img = Image.fromarray(cleaned)
    
    return processed_img

def post_process_ocr_text(text):
    """
    Post-process OCR text to fix common errors
    """
    if not text:
        return text
    
    # Common OCR character substitutions
    ocr_fixes = {
        '0': 'O',  # Zero to O in text context
        '1': 'l',  # One to lowercase l in text context
        '5': 'S',  # Five to S in text context
        '8': 'B',  # Eight to B in text context
        '6': 'G',  # Six to G in text context
        '9': 'g',  # Nine to g in text context
    }
    
    # Fix common OCR artifacts
    text = text.replace('|', 'I')  # Pipe to I
    text = text.replace('[]', 'o')  # Empty brackets to o
    text = text.replace('()', 'o')  # Empty parentheses to o
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text

def extract_pdf_with_ocr(pdf_path: str):
    """
    Extract text from scanned PDF pages using advanced OCR preprocessing.
    Returns a list of (book_page_number, text).
    """
    try:
        doc = fitz.open(pdf_path)
        page_texts = []
        total_pages = len(doc)
        
        print(f"Starting advanced OCR extraction from {total_pages} pages...")
        print("This may take several minutes for large PDFs...")

        for page_num, page in enumerate(doc, start=1):
            # Calculate book page number
            book_page = page_num - PAGE_OFFSET
            
            # Skip pages before content starts (negative book page numbers)
            if book_page <= 0:
                continue
                
            print(f"Processing PDF page {page_num} (Book page {book_page})/{total_pages}...", end="\r")
            
            # Render page as an image with very high resolution for better OCR
            pix = page.get_pixmap(dpi=600, alpha=False)  # Very high resolution
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # Advanced image preprocessing
            processed_img = preprocess_image_for_ocr(img)
            
            # Try multiple OCR configurations for best results
            ocr_configs = [
                r'--oem 3 --psm 6',  # Uniform block of text
                r'--oem 3 --psm 3',  # Fully automatic page segmentation
                r'--oem 3 --psm 4',  # Assume a single column of text
                r'--oem 3 --psm 1',  # Automatic page segmentation with OSD
            ]
            
            best_text = ""
            best_confidence = 0
            
            for config in ocr_configs:
                try:
                    # Get OCR result with confidence
                    data = pytesseract.image_to_data(processed_img, lang="eng", config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Get text
                    text = pytesseract.image_to_string(processed_img, lang="eng", config=config)
                    text = post_process_ocr_text(text)
                    
                    # Keep the result with highest confidence
                    if avg_confidence > best_confidence and text.strip():
                        best_text = text
                        best_confidence = avg_confidence
                        
                except Exception as e:
                    print(f"OCR config {config} failed: {e}")
                    continue
            
            if best_text.strip():
                page_texts.append((book_page, best_text.strip()))
            else:
                # Fallback: try with original image
                try:
                    fallback_text = pytesseract.image_to_string(img, lang="eng", config=r'--oem 3 --psm 6')
                    fallback_text = post_process_ocr_text(fallback_text)
                    if fallback_text.strip():
                        page_texts.append((book_page, fallback_text.strip()))
                except:
                    pass

        print(f"\nOCR extraction completed!")
        
        # Save all extracted text for processing
        if page_texts:
            with open("sample_extracted_text.txt", "w", encoding="utf-8") as f:
                for book_page, text in page_texts:  # Save ALL pages
                    f.write(f"=== BOOK PAGE {book_page} ===\n")
                    f.write(text)
                    f.write("\n\n" + "="*50 + "\n\n")
            print(f"Saved all {len(page_texts)} pages to 'sample_extracted_text.txt' for processing")
        
        return page_texts
        
    except pytesseract.pytesseract.TesseractNotFoundError:
        print("Tesseract OCR not found. Please install Tesseract OCR.")
        return []
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return []

# Setup Tesseract first
print("Setting up Tesseract OCR...")
tesseract_available = setup_tesseract()


# Use OCR as the only method
if tesseract_available:
    print("Using OCR to extract text from scanned PDF...")
    page_texts = extract_pdf_with_ocr(pdf_path)
    
    # Print comprehensive information about extracted text
    if page_texts:
        print(f"Successfully extracted text from {len(page_texts)} pages")
        print("=" * 60)
        
        # Show statistics
        total_chars = sum(len(text) for _, text in page_texts)
        avg_chars_per_page = total_chars // len(page_texts)
        print(f"Total characters extracted: {total_chars:,}")
        print(f"Average characters per page: {avg_chars_per_page:,}")
        print("=" * 60)
        
        # Show first few pages with more content
        print("Sample content from first few pages:")
        print("-" * 60)
        
        for i, (book_page, text) in enumerate(page_texts[:3]):  # Show first 3 pages
            print(f"\nBOOK PAGE {book_page} ({len(text)} characters):")
            print("-" * 40)
            # Show first 1000 characters of each page
            print(text[:1000])
            if len(text) > 1000:
                print(f"... (showing first 1000 of {len(text)} characters)")
            print("-" * 40)
        
        # Show a random page from the middle
        if len(page_texts) > 5:
            middle_page = len(page_texts) // 2
            book_page, text = page_texts[middle_page]
            print(f"\nSAMPLE FROM MIDDLE - BOOK PAGE {book_page} ({len(text)} characters):")
            print("-" * 40)
            print(text[:1000])
            if len(text) > 1000:
                print(f"... (showing first 1000 of {len(text)} characters)")
            print("-" * 40)
        
        # Show last page
        if len(page_texts) > 1:
            last_book_page, last_text = page_texts[-1]
            print(f"\nLAST PAGE - BOOK PAGE {last_book_page} ({len(last_text)} characters):")
            print("-" * 40)
            print(last_text[:1000])
            if len(last_text) > 1000:
                print(f"... (showing first 1000 of {len(last_text)} characters)")
            print("-" * 40)
            
    else:
        print("No text extracted from PDF")
else:
    print("Cannot proceed without Tesseract OCR. Please install it first.")