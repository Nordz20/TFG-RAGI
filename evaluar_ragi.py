import os
import csv
import json
import requests
import time
from elasticsearch import Elasticsearch

# ================== CONFIGURACIÓN ==================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "ragi_images"
EMBED_URL = os.getenv("EMBED_URL", "https://wiig.dia.fi.upm.es/ollama/api/embeddings")
EMBED_MODEL = "nomic-embed-text-v2-moe"
CSV_FILE = "ground_truth.csv"
CACHE_FILE = "cache_embeddings.json"
TOP_K = 5

# Cargar caché si existe
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(embedding_cache, f)

def get_embedding(text: str, retries=5) -> list[float]:
    # Si ya lo tenemos en caché, lo devolvemos inmediatamente
    if text in embedding_cache:
        return embedding_cache[text]

    payload = {"model": EMBED_MODEL, "prompt": text}
    
    for attempt in range(retries):
        try:
            response = requests.post(EMBED_URL, json=payload, timeout=120)
            response.raise_for_status()
            emb = response.json()["embedding"]
            # Guardar en caché
            embedding_cache[text] = emb
            save_cache()
            return emb
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"    [!] El servidor de la UPM rechazó la conexión. Esperando {wait_time}s... (Intento {attempt+2}/{retries})")
                time.sleep(wait_time)
            else:
                print("    [X] Fallo definitivo al conectar con el servidor de embeddings.")
                raise e

# ================== MODELOS DE BÚSQUEDA ==================

def search_baseline(es, query, top_k=TOP_K):
    response = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "match": {
                    "caption": query
                }
            },
            "size": top_k,
            "_source": ["image_path", "caption"]
        }
    )
    return [hit["_source"]["image_path"] for hit in response["hits"]["hits"]]

def search_ragi_basico(es, query, top_k=TOP_K):
    response = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["caption", "description"]
                }
            },
            "size": top_k,
            "_source": ["image_path", "caption"]
        }
    )
    return [hit["_source"]["image_path"] for hit in response["hits"]["hits"]]

def search_ragi_avanzado(es, query, top_k=TOP_K):
    query_embedding = get_embedding(query)
    response = es.search(
        index=ES_INDEX,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": 50
            },
            "_source": ["image_path", "caption"]
        }
    )
    return [hit["_source"]["image_path"] for hit in response["hits"]["hits"]]

# ================== EVALUACIÓN ==================

def evaluate():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print(f"❌ No se puede conectar a ElasticSearch en {ES_HOST}.")
        return

    print("[OK] Conectado a Elasticsearch. Iniciando evaluación...\n")

    queries = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append((row["Query"], row["Expected_Image"]))

    total_queries = len(queries)
    results = {
        "baseline": {"top1": 0, "top5": 0, "mrr": 0.0},
        "basico": {"top1": 0, "top5": 0, "mrr": 0.0},
        "avanzado": {"top1": 0, "top5": 0, "mrr": 0.0}
    }

    for idx, (query, expected_img) in enumerate(queries, 1):
        print(f"Evaluando ({idx}/{total_queries}): '{query}'")
        
        expected_img_clean = expected_img.replace('\\', '/').split('/data/')[-1]
        
        retrieved_baseline = search_baseline(es, query, top_k=TOP_K)
        retrieved_basico = search_ragi_basico(es, query, top_k=TOP_K)
        
        # Este intentará generar el embedding, o cargarlo de la caché
        retrieved_avanzado = search_ragi_avanzado(es, query, top_k=TOP_K)
        
        def calculate_metrics(retrieved, model_key):
            clean_retrieved = [p.replace('\\', '/').split('/data/')[-1] for p in retrieved]
            if expected_img_clean in clean_retrieved:
                rank = clean_retrieved.index(expected_img_clean) + 1
                if rank == 1:
                    results[model_key]["top1"] += 1
                if rank <= 5:
                    results[model_key]["top5"] += 1
                results[model_key]["mrr"] += (1.0 / rank)

        calculate_metrics(retrieved_baseline, "baseline")
        calculate_metrics(retrieved_basico, "basico")
        calculate_metrics(retrieved_avanzado, "avanzado")

    print("\n" + "="*50)
    print("📊 RESULTADOS FINALES DE LA EVALUACIÓN CUANTITATIVA")
    print("="*50)
    
    for model in ["baseline", "basico", "avanzado"]:
        top1_acc = (results[model]["top1"] / total_queries) * 100
        top5_acc = (results[model]["top5"] / total_queries) * 100
        mrr = results[model]["mrr"] / total_queries
        
        print(f"\nModelo: {model.upper()}")
        print(f"  - Precision@1 (Top-1): {top1_acc:.2f}%")
        print(f"  - Recall@5 (Top-5):    {top5_acc:.2f}%")
        print(f"  - MRR (Mean Rank):     {mrr:.4f}")

if __name__ == "__main__":
    evaluate()
