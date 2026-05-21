import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob

# Configuración de estilo global
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6)
})

OUTPUT_DIR = "graficas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = "analisis_ragi_votos.csv"
DATA_DIR = "data"

def cargar_metadatos():
    """Lee todos los manifest.json y crea un DataFrame con los metadatos de las imágenes."""
    archivos_manifest = glob.glob(os.path.join(DATA_DIR, "*", "manifest.json"))
    datos_imagenes = []
    
    for archivo in archivos_manifest:
        with open(archivo, 'r', encoding='utf-8') as f:
            try:
                manifest_data = json.load(f)
                for item in manifest_data:
                    # Extraer lo relevante
                    datos_imagenes.append({
                        'Ruta_Imagen': item.get('image_path'), # Coincide con la columna del CSV
                        'doc_id': item.get('doc_id'),
                        'page': item.get('page'),
                        'caption_length': len(str(item.get('caption', '')).split()),
                        'has_description2': 1 if item.get('description2') else 0
                    })
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")
                
    return pd.DataFrame(datos_imagenes)

def analizar_metadatos():
    print(f"Cargando votos desde {CSV_FILE}...")
    try:
        df_votos = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {CSV_FILE}")
        return

    print("Cargando metadatos de las imágenes...")
    df_meta = cargar_metadatos()
    
    # Hacer el cruce (JOIN) por Ruta_Imagen
    df = pd.merge(df_votos, df_meta, on='Ruta_Imagen', how='left')
    
    print(f"Filas tras el cruce: {len(df)}")
    if df['page'].isna().sum() > 0:
        print(f"Advertencia: {df['page'].isna().sum()} votos no encontraron metadatos.")

    # --- 7. Página del Documento vs Satisfacción ---
    plt.figure()
    sns.boxplot(data=df, x='Estrellas', y='page', palette="Purples")
    sns.stripplot(data=df, x='Estrellas', y='page', color=".25", alpha=0.5, jitter=True)
    plt.title("Página de la Imagen frente a la Valoración")
    plt.xlabel("Valoración (Estrellas)")
    plt.ylabel("Número de Página en el PDF")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig7_pagina_vs_satisfaccion.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 8. Longitud del Caption original vs Satisfacción ---
    plt.figure()
    sns.boxplot(data=df, x='Estrellas', y='caption_length', palette="Oranges")
    sns.stripplot(data=df, x='Estrellas', y='caption_length', color=".25", alpha=0.5, jitter=True)
    plt.title("Longitud del Caption Original frente a la Valoración")
    plt.xlabel("Valoración (Estrellas)")
    plt.ylabel("Nº Palabras del Caption de la Imagen")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_longitud_caption_vs_satisfaccion.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 9. Valoración Media por Paper (Top 10 Documentos más consultados) ---
    doc_stats = df.groupby('doc_id')['Estrellas'].agg(['mean', 'count']).reset_index()
    # Filtrar solo los documentos con al menos 2 valoraciones para que tenga sentido estadístico, o los top 10
    doc_stats_sorted = doc_stats[doc_stats['count'] >= 2].sort_values(by='mean', ascending=False)
    
    if not doc_stats_sorted.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=doc_stats_sorted.head(10), x='mean', y='doc_id', palette="magma")
        plt.title("Valoración Media por Documento (Paper)")
        plt.xlabel("Valoración Media (Estrellas)")
        plt.ylabel("ID del Documento (arXiv)")
        plt.xlim(0, 5.5)
        
        # Añadir el número de veces que se votó en la barra
        for index, row in enumerate(doc_stats_sorted.head(10).itertuples()):
            plt.text(row.mean + 0.1, index, f"n={row.count}", color='black', va="center")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fig9_valoracion_por_paper.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No hay suficientes datos agrupados por documento para generar la figura 9.")

    print(f"✅ Se han generado las nuevas gráficas cruzadas en la carpeta '{OUTPUT_DIR}/'.")
    
    # Datos curiosos para redactar:
    correlacion_caption = df['Estrellas'].corr(df['caption_length'])
    correlacion_pagina = df['Estrellas'].corr(df['page'])
    
    print("\n--- INSIGHTS DE METADATOS ---")
    print(f"Correlación (Pearson) Estrellas vs Longitud del Caption: {correlacion_caption:.3f}")
    print(f"Correlación (Pearson) Estrellas vs Página del PDF: {correlacion_pagina:.3f}")

if __name__ == "__main__":
    analizar_metadatos()
