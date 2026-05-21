import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import os

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (12, 8)})
OUTPUT_DIR = "graficas"

CSV_FILE = "analisis_ragi_votos.csv"
CACHE_FILE = "cache_embeddings.json"

def generar_grafica_embeddings_local():
    df = pd.read_csv(CSV_FILE)
    queries_unicas = df['Query'].unique()
    print(f"Buscando {len(queries_unicas)} queries en la caché local...")
    
    if not os.path.exists(CACHE_FILE):
        print("El archivo de caché no existe.")
        return
        
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
        
    embeddings = []
    textos_validos = []
    
    for q in queries_unicas:
        if q in embedding_cache:
            embeddings.append(embedding_cache[q])
            textos_validos.append(str(q))
        else:
            print(f"Falta en caché: {q}")
            
    if not embeddings:
        print("No se encontraron embeddings en la caché para las queries.")
        return
        
    print(f"Procesando {len(embeddings)} vectores...")
    X = np.array(embeddings)
    
    num_clusters = min(4, len(X) // 2)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    df_plot = pd.DataFrame({
        'PCA1': X_2d[:, 0],
        'PCA2': X_2d[:, 1],
        'Cluster': [f"Intención {c+1}" for c in clusters],
        'Query': textos_validos
    })
    
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
    
    for i in range(len(df_plot)):
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
    generar_grafica_embeddings_local()