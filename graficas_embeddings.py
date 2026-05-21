import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import json
import requests
import time
import os

# Configuración de estilo global
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (12, 8)})
OUTPUT_DIR = "graficas"

# Archivos y endpoints
CSV_FILE = "analisis_ragi_votos.csv"
CACHE_FILE = "cache_embeddings.json"
EMBED_URL = "https://wiig.dia.fi.upm.es/ollama/api/embeddings"
EMBED_MODEL = "nomic-embed-text-v2-moe"

# Cargar caché si existe
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(embedding_cache, f)

def get_embedding(text: str, retries=5):
    if text in embedding_cache:
        return embedding_cache[text]

    payload = {"model": EMBED_MODEL, "prompt": text}
    
    for attempt in range(retries):
        try:
            response = requests.post(EMBED_URL, json=payload, timeout=30)
            response.raise_for_status()
            emb = response.json()["embedding"]
            embedding_cache[text] = emb
            save_cache()
            return emb
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 * (attempt + 1)
                print(f"    [!] Retrying... ({attempt+2}/{retries}) - {text[:30]}...")
                time.sleep(wait_time)
            else:
                print(f"    [X] Failed getting embedding for: {text}")
                return None

def generar_grafica_embeddings():
    df = pd.read_csv(CSV_FILE)
    
    # Obtener queries únicas para no repetir llamadas
    queries_unicas = df['Query'].unique()
    print(f"Obteniendo embeddings para {len(queries_unicas)} queries únicas...")
    
    embeddings = []
    textos_validos = []
    
    for q in queries_unicas:
        emb = get_embedding(str(q))
        if emb is not None:
            embeddings.append(emb)
            textos_validos.append(str(q))
            
    if not embeddings:
        print("No se pudieron obtener embeddings.")
        return
        
    X = np.array(embeddings)
    
    # Reducción de dimensionalidad: PCA o t-SNE
    # Como son pocas queries, usamos PCA para ser estables, o t-SNE con perplexity baja
    n_samples = len(X)
    print(f"Reduciendo dimensiones de {n_samples} vectores...")
    
    # Agrupamiento por K-Means para encontrar "Intenciones de Búsqueda"
    # Buscamos por ejemplo 3 clústeres lógicos (ej. Redes neuronales, gráficos, diagramas de flujo)
    num_clusters = min(4, n_samples // 2)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Aplicar PCA a 2 componentes
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    # Crear un DataFrame para plotear
    df_plot = pd.DataFrame({
        'PCA1': X_2d[:, 0],
        'PCA2': X_2d[:, 1],
        'Cluster': [f"Intención {c+1}" for c in clusters],
        'Query': textos_validos
    })
    
    # Generar Scatter Plot
    plt.figure()
    sns.scatterplot(
        data=df_plot, 
        x='PCA1', 
        y='PCA2', 
        hue='Cluster', 
        palette='tab10', 
        s=150, 
        alpha=0.8,
        edgecolor='k'
    )
    
    # Añadir etiquetas de texto (sólo algunas para no saturar si hay muchas superpuestas, pero como son pocas queries, ponemos todas o acortadas)
    for i in range(len(df_plot)):
        # Acortar la query para que quepa en el gráfico
        texto = df_plot['Query'].iloc[i]
        if len(texto) > 25:
            texto = texto[:22] + "..."
            
        plt.text(df_plot['PCA1'].iloc[i] + 0.02, df_plot['PCA2'].iloc[i] + 0.02, 
                 texto, fontsize=8, alpha=0.7)

    plt.title(f"Mapa Semántico de Consultas (PCA sobre Embeddings de Nomic)\n{num_clusters} Clústeres de Intención Detectados")
    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
    plt.legend(title="Grupo de Intención", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig12_clusters_embeddings.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfica 12 generada: Mapa Semántico de Consultas (PCA).")

if __name__ == "__main__":
    generar_grafica_embeddings()
