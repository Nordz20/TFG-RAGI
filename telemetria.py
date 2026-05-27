import csv
from elasticsearch import Elasticsearch
import os
import random

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "ragi_images"
VOTOS_FILE = "analisis_ragi_votos.csv"
TOP_K = 5

def search_bm25(es, query, top_k=TOP_K):
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
            "_source": ["image_path"]
        }
    )
    return [hit["_source"]["image_path"].replace('\\', '/').split('/data/')[-1] for hit in response["hits"]["hits"]]

def evaluate_offline():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("❌ No se puede conectar a ElasticSearch.")
        return

    # 1. Filtrar consultas reales de usuarios con 4 o 5 estrellas
    high_rating_queries = []
    with open(VOTOS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Estrellas"]) >= 4:
                # Normalizar ruta
                img_path = row["Ruta_Imagen"].replace('\\', '/').split('/data/')[-1]
                high_rating_queries.append({"query": row["Query"], "expected": img_path})

    # Tomar una muestra representativa (ej. 20) si hay muchas, o todas si hay menos.
    random.seed(42)
    sample_size = min(20, len(high_rating_queries))
    sample = random.sample(high_rating_queries, sample_size)

    print(f"==================================================")
    print(f"📊 EVALUACIÓN OFFLINE DE TELEMETRÍA (N={sample_size})")
    print(f"==================================================\n")

    top1_bm25 = 0
    top5_bm25 = 0

    for item in sample:
        query = item["query"]
        expected = item["expected"]
        
        results = search_bm25(es, query)
        
        if expected in results:
            rank = results.index(expected) + 1
            if rank == 1:
                top1_bm25 += 1
            if rank <= 5:
                top5_bm25 += 1
                
    print(f"Resultados de BM25 simulando consultas humanas reales:")
    print(f"  - Precision@1: {(top1_bm25/sample_size)*100:.2f}%")
    print(f"  - Recall@5:    {(top5_bm25/sample_size)*100:.2f}%\n")
    print("Nota: El modelo semántico encontró estas imágenes en Top-5 durante el uso de la app (por eso tienen 4/5 estrellas).")

if __name__ == "__main__":
    evaluate_offline()