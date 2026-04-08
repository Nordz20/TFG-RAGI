import json
import time
from pathlib import Path

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan  # NUEVO: Necesario para leer todos los IDs eficientemente

# ================== CONFIG ==================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# ElasticSearch
ES_HOST = "http://localhost:9200"
ES_INDEX = "ragi_images"

# Modelo de embeddings (servidor universidad)
EMBED_URL = "https://wiig.dia.fi.upm.es/ollama/api/embeddings"
EMBED_MODEL = "nomic-embed-text-v2-moe"  # cambiar cuando esté disponible

SLEEP_BETWEEN = 0.2
FORCE_REINDEX = False  # True para reindexar aunque ya exista el doc


# ================== ELASTICSEARCH SETUP ==================
def create_index(es: Elasticsearch, dim: int):
    """Crea el índice con mapping para búsqueda semántica (kNN)."""
    mapping = {
        "mappings": {
            "properties": {
                "doc_id":       {"type": "keyword"},
                "source_pdf":   {"type": "keyword"},
                "source_pdf_url": {"type": "keyword"},
                "page":         {"type": "integer"},
                "image_path":   {"type": "keyword"},
                "caption":      {"type": "text"},
                "description2": {"type": "text"},
                "description3": {"type": "text"},
                "questions":    {"type": "text"},
                "full_text":    {"type": "text"},   # texto combinado
                "embedding":    {                   # vector para kNN
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    if es.indices.exists(index=ES_INDEX):
        print(f"[INFO] Índice '{ES_INDEX}' ya existe, se omite creación.")
    else:
        es.indices.create(index=ES_INDEX, body=mapping)
        print(f"[OK] Índice '{ES_INDEX}' creado con dim={dim}.")


# ================== TEXTO COMBINADO ==================
def build_full_text(entry: dict) -> str:
    """Combina caption + description2 + description3 + questions en un solo texto."""
    parts = []

    caption = entry.get("caption", "").strip()
    if caption:
        parts.append(f"Caption: {caption}")

    desc2 = entry.get("description2", "").strip()
    if desc2:
        parts.append(f"Description: {desc2}")

    desc3 = entry.get("description3", "").strip()
    if desc3:
        parts.append(f"Description: {desc3}")

    questions = entry.get("questions", [])
    if questions:
        q_text = " | ".join(q.strip() for q in questions if q.strip())
        parts.append(f"Questions: {q_text}")

    return " ".join(parts)


# ================== EMBEDDINGS ==================
def get_embedding(text: str) -> list[float]:
    """Llama al modelo de embeddings del servidor de la universidad."""
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    response = requests.post(EMBED_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["embedding"]


# ================== INDEXACIÓN ==================
def index_document(es: Elasticsearch, entry: dict, embedding: list[float]):
    """Indexa un documento en ElasticSearch."""
    doc_id = f"{entry['doc_id']}_{entry['id']}"

    # Comprobar si ya existe
    if not FORCE_REINDEX and es.exists(index=ES_INDEX, id=doc_id):
        print(f"  [SKIP] {doc_id} ya indexado.")
        return

    doc = {
        "doc_id":         entry.get("doc_id"),
        "source_pdf":     entry.get("source_pdf"),
        "source_pdf_url": entry.get("source_pdf_url"),
        "page":           entry.get("page"),
        "image_path":     entry.get("image_path"),
        "caption":        entry.get("caption", ""),
        "description2":   entry.get("description2", ""),
        "description3":   entry.get("description3", ""),
        "questions":      entry.get("questions", []),
        "full_text":      entry.get("_full_text", ""),
        "embedding":      embedding
    }

    es.index(index=ES_INDEX, id=doc_id, document=doc)
    print(f"  [OK] {doc_id} indexado.")


# ================== MAIN ==================
def main():
    # 1. Conectar a ElasticSearch
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError(f"No se puede conectar a ElasticSearch en {ES_HOST}")
    print(f"[OK] Conectado a ElasticSearch en {ES_HOST}")

    # 2. Leer todos los manifests
    manifests = sorted(DATA_DIR.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No se encontraron manifests en {DATA_DIR}")
    print(f"[INFO] {len(manifests)} manifests encontrados.")

    # 3. Calcular textos y recopilar IDs locales
    all_entries = []
    local_ids = set()
    for manifest_path in manifests:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            entry["_full_text"] = build_full_text(entry)
            all_entries.append(entry)
            local_ids.add(f"{entry['doc_id']}_{entry['id']}")

    print(f"[INFO] Se han cargado {len(all_entries)} imágenes locales listas para procesar.")

    # 4. Obtener dimensión del embedding (necesario para crear el índice si no existe)
    print("[INFO] Obteniendo dimensión del embedding de prueba...")
    sample_embedding = get_embedding(all_entries[0]["_full_text"])
    dim = len(sample_embedding)
    
    # 5. Crear el índice ANTES de limpiar (por si es la primera vez que se ejecuta)
    create_index(es, dim)

    # 6. Limpiar documentos huérfanos en Elasticsearch
    print("[INFO] Buscando documentos huérfanos en Elasticsearch...")
    es_docs = scan(es, index=ES_INDEX, query={"_source": False})
    es_ids = {doc["_id"] for doc in es_docs}
    
    orphans = es_ids - local_ids
    if orphans:
        print(f"[INFO] Se encontraron {len(orphans)} documentos huérfanos. Borrando...")
        for orphan_id in orphans:
            try:
                es.delete(index=ES_INDEX, id=orphan_id)
            except Exception as e:
                print(f"  [ERROR] No se pudo borrar {orphan_id}: {e}")
    else:
        print("[INFO] La base de datos está limpia, no hay huérfanos.")

    # 7. Forzar actualización de Elasticsearch y contar cuántos hay antes de empezar
    es.indices.refresh(index=ES_INDEX)
    initial_count = es.count(index=ES_INDEX)['count']
    
    if initial_count == 0:
        print("\n[ESTADO] -> Actualmente NO hay ningún PDF/imagen indexado en la base de datos. Está vacía.\n")
    else:
        print(f"\n[ESTADO] -> Actualmente hay {initial_count} PDFs/imágenes indexados en la base de datos.\n")

    # 8. Indexar todos los documentos
    print("[INFO] Iniciando proceso de indexación...")
    for i, entry in enumerate(all_entries):
        print(f"[{i+1}/{len(all_entries)}] {entry['doc_id']} - imagen {entry['id']}")
        try:
            embedding = get_embedding(entry["_full_text"])
            index_document(es, entry, embedding)
        except Exception as e:
            print(f"  [ERROR] {e}")
        time.sleep(SLEEP_BETWEEN)

    # 9. Forzar actualización final y contar el total
    es.indices.refresh(index=ES_INDEX)
    final_count = es.count(index=ES_INDEX)['count']

    print(f"\n{'='*50}")
    print(f"[DONE] Indexación completada con éxito.")
    print(f"[RESULTADO FINAL] Hay un TOTAL de {final_count} PDFs/imágenes indexados en Elasticsearch.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()