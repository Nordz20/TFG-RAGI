import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (12, 8)})
OUTPUT_DIR = "graficas"
CSV_FILE = "analisis_ragi_votos.csv"

def generar_grafica_embeddings_local_huggingface():
    df = pd.read_csv(CSV_FILE)
    queries_unicas = df['Query'].unique()
    print(f"Calculando embeddings localmente para {len(queries_unicas)} queries únicas usando sentence-transformers (CPU mode)...")
    
    # Force CPU to avoid CUDA error with incompatible GPU
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    textos_validos = [str(q) for q in queries_unicas]
    embeddings = model.encode(textos_validos)
    
    X = np.array(embeddings)
    
    # Agrupamiento por K-Means
    num_clusters = min(4, len(X) // 2)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # PCA
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
    
    # Etiquetas
    for i in range(len(df_plot)):
        texto = df_plot['Query'].iloc[i]
        if len(texto) > 25:
            texto = texto[:22] + "..."
        plt.text(df_plot['PCA1'].iloc[i] + 0.02, df_plot['PCA2'].iloc[i] + 0.02, 
                 texto, fontsize=8, alpha=0.7)

    plt.title(f"Mapa Semántico de Consultas (PCA sobre Embeddings de MiniLM)\n{num_clusters} Clústeres de Intención Detectados")
    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
    plt.legend(title="Grupo de Intención", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig12_clusters_embeddings.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfica 12 generada: Mapa Semántico de Consultas (PCA).")

if __name__ == "__main__":
    generar_grafica_embeddings_local_huggingface()
