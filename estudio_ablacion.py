import os
import csv
from elasticsearch import Elasticsearch

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "ragi_images"
CSV_FILE = "ground_truth.csv"
TOP_K = 5

def search_fields(es, query, fields, top_k=TOP_K):
    response = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields
                }
            },
            "size": top_k,
            "_source": ["image_path"]
        }
    )
    return [hit["_source"]["image_path"] for hit in response["hits"]["hits"]]

def evaluate():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print(f"❌ No se puede conectar a ElasticSearch en {ES_HOST}.")
        return

    queries = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append((row["Query"], row["Expected_Image"]))

    combinations = {
        "Caption solo": ["caption"],
        "Caption + Descripcion": ["caption", "description"],
        "Caption + Preguntas": ["caption", "questions"],
        "Caption + Descripcion + Preguntas": ["caption", "description", "questions"]
    }

    total = len(queries)
    print("==================================================")
    print("📊 RESULTADOS DEL ESTUDIO DE ABLACIÓN (Campos Léxicos)")
    print("==================================================\n")

    for name, fields in combinations.items():
        top1 = 0
        top5 = 0
        mrr = 0.0
        for query, expected in queries:
            expected_clean = expected.replace('\\', '/').split('/data/')[-1]
            results = search_fields(es, query, fields)
            clean_results = [p.replace('\\', '/').split('/data/')[-1] for p in results]
            
            if expected_clean in clean_results:
                rank = clean_results.index(expected_clean) + 1
                if rank == 1: top1 += 1
                if rank <= 5: top5 += 1
                mrr += 1.0 / rank
        
        print(f"Modelo: {name} (Campos: {fields})")
        print(f"  - Precision@1: {(top1/total)*100:.2f}%")
        print(f"  - Recall@5:    {(top5/total)*100:.2f}%")
        print(f"  - MRR:         {mrr/total:.4f}\n")

if __name__ == '__main__':
    evaluate()
