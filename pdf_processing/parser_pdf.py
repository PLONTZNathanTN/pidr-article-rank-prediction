import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
import sys

# Add the root directory of the project (where the DB folder is) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def count_images_in_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    image_count = 0

    for page in doc:
        images = page.get_images(full=True)
        image_count += len(images)

    return image_count

def extract_pdf_info(pdf_path):
    """Extracts information from a PDF: number of characters, pages, and lines."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        num_pages = len(doc)
        
        for page in doc:
            text += page.get_text("text")
        
        num_chars = len(text)
        num_lines = text.count("\n") + 1  # Count lines

        return {
            "num_chars": num_chars,
            "num_pages": num_pages,
            "num_lines": num_lines,
            "num_image": count_images_in_pdf(pdf_path)
        }
    except Exception as e:
        print(f"Error with {pdf_path}: {e}")
        return None

def process_pdfs(input_dir, output_json):
    """Recursively scans input_dir, analyzes all PDFs and saves results in a JSON file."""
    results = {}

    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    with tqdm(total=len(pdf_files), desc="Analyzing PDFs", unit="pdf") as pbar:
        for pdf_path in pdf_files:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]  # File name without extension
            info = extract_pdf_info(pdf_path)
            if info:
                results[pdf_name] = info
            pbar.update(1)

    # Save as JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Analysis complete. Results saved to {output_json}")

def process_limited_pdfs(input_dir, output_json, limit):
    """Scans input_dir, analyzes a limited number of PDFs, and saves the results."""
    results = {}

    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    
    pdf_files = pdf_files[:limit]  # Take only the first 'limit' files

    with tqdm(total=len(pdf_files), desc=f"Analyzing {limit} PDFs", unit="pdf") as pbar:
        for pdf_path in pdf_files:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]  # File name without extension
            info = extract_pdf_info(pdf_path)
            if info:
                results[pdf_name] = info
            pbar.update(1)

    # Save as JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Analysis of {limit} PDFs complete. Results saved to {output_json}")

def verify_pdf_analysis(input_dir, json_file):
    """Checks if all PDFs in the folder are present in the JSON file."""
    # Load analysis results from JSON file
    if not os.path.exists(json_file):
        print(f"JSON file {json_file} not found. Please run the analysis first.")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # List of analyzed PDFs
    analyzed_pdfs = set(data.keys())

    # List of all PDFs in the directory
    all_pdfs = set()
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_name = os.path.splitext(file)[0]
                all_pdfs.add(pdf_name)

    # Compare the lists
    missing_pdfs = all_pdfs - analyzed_pdfs  # PDFs not in JSON
    extra_pdfs = analyzed_pdfs - all_pdfs    # PDFs in JSON but not in the folder

    if missing_pdfs:
        print("\n MISSING PDFs IN JSON:")
        for pdf in missing_pdfs:
            print(f"- {pdf}.pdf")

    if extra_pdfs:
        print("\n PDFs RECORDED IN JSON BUT NOT FOUND IN FOLDER:")
        for pdf in extra_pdfs:
            print(f"- {pdf}.pdf (exists in {json_file} but not in {input_dir})")

    if not missing_pdfs and not extra_pdfs:
        print("\n All PDFs are correctly recorded in the JSON file.")
