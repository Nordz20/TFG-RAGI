import os
import csv
import json
import requests
from elasticsearch import Elasticsearch

# Configuración basada en tu evaluar_ragi.py
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "ragi_images"
EMBED_URL = os.getenv("EMBED_URL", "https://wiig.dia.fi.upm.es/ollama/api/embeddings")
EMBED_MODEL = "nomic-embed-text-v2-moe"
CSV_FILE = "ground_truth.csv"
CACHE_FILE = "cache_embeddings.json"

# Cargar caché si existe para ir más rápido
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

def get_embedding(text: str) -> list[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(EMBED_URL, json=payload, timeout=120)
    response.raise_for_status()
    emb = response.json()["embedding"]
    embedding_cache[text] = emb
    with open(CACHE_FILE, "w") as f:
        json.dump(embedding_cache, f)
    return emb

def search_lexico(es, query):
    # BM25 sobre caption y description (basico)
    response = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["caption", "description"]
                }
            },
            "size": 1,
            "_source": ["image_path", "caption"]
        }
    )
    hits = response["hits"]["hits"]
    if hits:
        return hits[0]["_source"]["image_path"].replace('\\', '/').split('/data/')[-1]
    return None

def search_semantico(es, query):
    # kNN sobre embeddings (avanzado)
    query_embedding = get_embedding(query)
    response = es.search(
        index=ES_INDEX,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": 1,
                "num_candidates": 50
            },
            "_source": ["image_path", "caption"]
        }
    )
    hits = response["hits"]["hits"]
    if hits:
        return hits[0]["_source"]["image_path"].replace('\\', '/').split('/data/')[-1]
    return None

def main():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print(f"❌ No se puede conectar a ElasticSearch en {ES_HOST}.")
        return

    print("🔎 Buscando divergencias (Léxico acierta, Semántico falla)...\n")
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["Query"]
            expected = row["Expected_Image"].replace('\\', '/').split('/data/')[-1]
            
            top1_lexico = search_lexico(es, query)
            top1_semantico = search_semantico(es, query)
            
            # Condición de divergencia: Léxico acierta y Semántico falla
            if top1_lexico == expected and top1_semantico != expected:
                print(f"🔹 Query: '{query}'")
                print(f"   ✔️ Léxico acertó: {top1_lexico}")
                print(f"   ❌ Semántico falló. Devolvió: {top1_semantico}")
                print("-" * 50)

if __name__ == "__main__":
    main()