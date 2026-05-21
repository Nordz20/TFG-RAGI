import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo global para que todas queden consistentes y académicas
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6)
})

# Crear directorio para guardar las gráficas si no existe
OUTPUT_DIR = "graficas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = "analisis_ragi_votos.csv"

def generar_graficas():
    print(f"Cargando datos desde {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {CSV_FILE}")
        return

    # Preparar datos adicionales
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Longitud_Query'] = df['Query'].apply(lambda x: len(str(x).split()))
    
    # Categorizar satisfacción
    def categorizar_satisfaccion(estrellas):
        if estrellas <= 2: return 'Baja (1-2)'
        elif estrellas == 3: return 'Neutra (3)'
        else: return 'Alta (4-5)'
    df['Nivel_Satisfaccion'] = df['Estrellas'].apply(categorizar_satisfaccion)

    # Colores base
    color_palette = "viridis"

    # --- 1. Distribución de Estrellas (Histograma/Barras) ---
    plt.figure()
    ax = sns.countplot(data=df, x='Estrellas', palette=color_palette)
    plt.title("Distribución de Valoraciones de los Usuarios")
    plt.xlabel("Valoración (Estrellas)")
    plt.ylabel("Frecuencia (Nº de votos)")
    # Añadir números sobre las barras
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_distribucion_estrellas.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. Proporción de Satisfacción (Gráfico de Anillo) ---
    plt.figure()
    counts = df['Nivel_Satisfaccion'].value_counts()
    colores_anillo = ['#2ca02c', '#d62728', '#ff7f0e'] # Verde (Alta), Rojo (Baja), Naranja (Neutra) - Ajustar según orden
    # Reordenar para consistencia visual si es necesario
    counts = counts.reindex(['Alta (4-5)', 'Neutra (3)', 'Baja (1-2)']).fillna(0)
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#66c2a5', '#fc8d62', '#8da0cb'], pctdistance=0.85)
    # Dibujar círculo en el medio para hacer el anillo
    centro_circulo = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centro_circulo)
    plt.title("Proporción de Niveles de Satisfacción")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_satisfaccion_agrupada.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. Longitud de la Consulta vs. Satisfacción (Dispersión/Boxplot) ---
    plt.figure()
    sns.boxplot(data=df, x='Estrellas', y='Longitud_Query', palette="Blues")
    sns.stripplot(data=df, x='Estrellas', y='Longitud_Query', color=".25", alpha=0.5, jitter=True)
    plt.title("Longitud de la Consulta frente a la Valoración")
    plt.xlabel("Valoración (Estrellas)")
    plt.ylabel("Número de palabras en la consulta")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_longitud_vs_satisfaccion.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Calcular medias por consulta para Top Mejores/Peores
    query_stats = df.groupby('Query')['Estrellas'].agg(['mean', 'count']).reset_index()
    # Filtrar queries que tengan al menos un número mínimo de votos si es necesario, o cogerlas todas
    # Ordenar por media
    query_stats_sorted = query_stats.sort_values(by='mean', ascending=False)

    # --- 4. Top Mejores Consultas (Barras Horizontales) ---
    plt.figure(figsize=(10, 5))
    top_mejores = query_stats_sorted.head(5)
    sns.barplot(data=top_mejores, x='mean', y='Query', palette="Greens_r")
    plt.title("Top 5 Consultas con Mejor Valoración Media")
    plt.xlabel("Valoración Media (Estrellas)")
    plt.ylabel("")
    plt.xlim(0, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_top_mejores_consultas.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 5. Top Peores Consultas (Barras Horizontales) ---
    plt.figure(figsize=(10, 5))
    top_peores = query_stats_sorted.tail(5).sort_values(by='mean', ascending=True)
    sns.barplot(data=top_peores, x='mean', y='Query', palette="Reds_r")
    plt.title("Top 5 Consultas con Peor Valoración Media")
    plt.xlabel("Valoración Media (Estrellas)")
    plt.ylabel("")
    plt.xlim(0, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_top_peores_consultas.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 6. Evolución Temporal (Línea de tendencia) ---
    plt.figure()
    df_temporal = df.sort_values(by='Timestamp').copy()
    # Calcular media móvil para suavizar (ventana pequeña por ser pocos datos)
    df_temporal['Media_Movil_5'] = df_temporal['Estrellas'].rolling(window=5, min_periods=1).mean()
    
    plt.plot(df_temporal['Timestamp'], df_temporal['Estrellas'], marker='o', linestyle='', alpha=0.3, label='Votos individuales')
    plt.plot(df_temporal['Timestamp'], df_temporal['Media_Movil_5'], color='red', linewidth=2, label='Media Móvil (n=5)')
    
    plt.title("Evolución de las Valoraciones a lo Largo de la Sesión")
    plt.xlabel("Tiempo")
    plt.ylabel("Valoración (Estrellas)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_evolucion_temporal.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Se han generado las 6 gráficas en formato PNG en la carpeta '{OUTPUT_DIR}/'.")
    
    # Calcular Media Global (MOS)
    mos = df['Estrellas'].mean()
    print(f"\n--- ESTADÍSTICAS RÁPIDAS ---")
    print(f"Total de valoraciones: {len(df)}")
    print(f"Media Global (Mean Opinion Score - MOS): {mos:.2f} sobre 5.00")

if __name__ == "__main__":
    generar_graficas()
