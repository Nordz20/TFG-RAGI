import os
import json
import csv
import re
import random

random.seed(123)

DATA_DIR = "data"
OUTPUT_CSV = "ground_truth.csv"

# Stopwords y frases de inicio a eliminar
REMOVE_PREFIXES = [
    r'^(?:fig\.|figure)\s*[\d\w\(\),]+\s*:\s*',
    r'^(?:the\s+|a\s+|an\s+)?(?:illustration|overview|schematic|diagram|flowchart|plot|graph|table|example|comparison|timeline)\s+(?:of|showing|depicting|illustrating|comparing)\s+',
    r'^(?:the\s+|a\s+|an\s+)?figure\s+(?:shows|illustrates|depicts|presents)\s+'
]

def clean_to_query(text):
    # 1. Limpiar prefijos de figuras y frases introductorias
    for prefix in REMOVE_PREFIXES:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # 2. Coger la primera frase útil (antes de punto o punto y coma)
    text = re.split(r'[.;]', text)[0]
    
    # 3. Eliminar referencias a citas tipo [14], [23]
    text = re.sub(r'\[\d+\]', '', text)
    
    # 4. Eliminar lo que esté entre paréntesis
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 5. Limpieza de espacios y comillas
    text = text.replace('"', '').replace("'", "").strip()
    
    # 6. Quedarse con las primeras palabras clave (entre 3 y 7 palabras) para simular un usuario
    words = text.split()
    if len(words) > 6:
        query = " ".join(words[:random.randint(4, 6)])
    else:
        query = " ".join(words)
        
    return query.lower()

def main():
    all_images = []
    
    for root, dirs, files in os.walk(DATA_DIR):
        if "manifest.json" in files:
            manifest_path = os.path.join(root, "manifest.json")
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        caption = item.get("caption", "")
                        # Solo coger imágenes que tengan un caption decentemente largo (evitar los que son solo "Fig 1.")
                        if len(caption) > 40:
                            all_images.append(item)
            except Exception:
                pass
                
    # Coger 50 imágenes aleatorias con caption útil
    selected_images = random.sample(all_images, min(50, len(all_images)))
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Expected_Image"])
        
        valid_count = 0
        for item in selected_images:
            image_path = item.get("image_path", "")
            caption = item.get("caption", "")
            
            # Intentar generar el prompt realista
            query = clean_to_query(caption)
            
            # Filtro de calidad final: que la query tenga al menos 3 palabras
            if len(query.split()) >= 3:
                writer.writerow([query, image_path])
                valid_count += 1
                
    print(f"Se ha creado {OUTPUT_CSV} con {valid_count} prompts realistas (simulando búsquedas de usuarios).")

if __name__ == "__main__":
    main()
