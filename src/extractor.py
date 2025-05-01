import fitz  # PyMuPDF
import os

def extract_texts_from_pdfs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            output_path = os.path.join(output_folder, filename.replace('.pdf', '.txt'))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)