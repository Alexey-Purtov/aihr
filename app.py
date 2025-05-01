from src.extractor import extract_texts_from_pdfs
from src.cleaner import clean_text
from src.matcher import match_new_to_base
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# === Параметры ===
base_pdf_folder = 'data/base_resumes'
new_pdf_folder = 'data/resumes'
base_text_folder = 'extracted/base_texts'
new_text_folder = 'extracted/texts'
embedding_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# === 1. Извлечение текста из PDF ===
extract_texts_from_pdfs(base_pdf_folder, base_text_folder)
extract_texts_from_pdfs(new_pdf_folder, new_text_folder)

# === 2. Чтение и очистка текстов ===
base_texts = {}
for filename in os.listdir(base_text_folder):
    if filename.startswith('.') or not filename.endswith('.txt'):  # Пропускаем скрытые файлы и не текстовые файлы
        continue
    try:
        with open(os.path.join(base_text_folder, filename), 'r', encoding='utf-8') as f:
            base_texts[filename] = clean_text(f.read())
    except UnicodeDecodeError:
        try:
            with open(os.path.join(base_text_folder, filename), 'r', encoding='cp1251') as f:
                base_texts[filename] = clean_text(f.read())
        except UnicodeDecodeError:
            with open(os.path.join(base_text_folder, filename), 'r', encoding='utf-8', errors='ignore') as f:
                base_texts[filename] = clean_text(f.read())

new_texts = {}
for filename in os.listdir(new_text_folder):
    if filename.startswith('.') or not filename.endswith('.txt'):  # Пропускаем скрытые файлы и не текстовые файлы
        continue
    try:
        with open(os.path.join(new_text_folder, filename), 'r', encoding='utf-8') as f:
            new_texts[filename] = clean_text(f.read())
    except UnicodeDecodeError:
        try:
            with open(os.path.join(new_text_folder, filename), 'r', encoding='cp1251') as f:
                new_texts[filename] = clean_text(f.read())
        except UnicodeDecodeError:
            with open(os.path.join(new_text_folder, filename), 'r', encoding='utf-8', errors='ignore') as f:
                new_texts[filename] = clean_text(f.read())

# === 3. Векторизация текстов ===
model = SentenceTransformer(embedding_model_name)

base_vectors = model.encode(list(base_texts.values()))
new_vectors = model.encode(list(new_texts.values()))

# === 4. Подсчет похожести ===
results = match_new_to_base(new_texts.keys(), new_vectors, base_vectors)

# === 5. Вывод результатов ===
df = pd.DataFrame(results, columns=["Резюме", "Средняя похожесть (%)"])
df = df.sort_values(by="Средняя похожесть (%)", ascending=False)
print("\nРезультаты:\n")
print(df)

# Сохраняем
df.to_csv("matching_results.csv", index=False)