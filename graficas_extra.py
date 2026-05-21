import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

OUTPUT_DIR = "graficas"
CSV_FILE = "analisis_ragi_votos.csv"

# Configuración
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6)})

def generar_nuevas_graficas():
    df = pd.read_csv(CSV_FILE)
    
    # 1. Frecuencia de palabras (excluyendo stop words básicas)
    stop_words = {'show', 'me', 'an', 'image', 'about', 'of', 'a', 'the', 'with', 'in', 'and', 'to', 'for', 'on'}
    todas_palabras = []
    
    # Diferenciar tipos de consulta
    df['Es_Conversacional'] = df['Query'].str.lower().str.contains('show me|find|give me|i want')
    
    for query in df['Query']:
        palabras = re.findall(r'\b\w+\b', str(query).lower())
        palabras_limpias = [p for p in palabras if p not in stop_words]
        todas_palabras.extend(palabras_limpias)
        
    conteo = Counter(todas_palabras)
    top_palabras = pd.DataFrame(conteo.most_common(10), columns=['Palabra', 'Frecuencia'])
    
    # --- 10. Top Palabras Clave ---
    plt.figure()
    sns.barplot(data=top_palabras, x='Frecuencia', y='Palabra', palette='Blues_r')
    plt.title("Top 10 Conceptos Más Buscados en el Estudio")
    plt.xlabel("Frecuencia de aparición en las consultas")
    plt.ylabel("")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_top_palabras_clave.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 11. Tipo de Consulta (Conversacional vs Keywords) vs Satisfacción ---
    plt.figure()
    sns.boxplot(data=df, x='Es_Conversacional', y='Estrellas', palette='Set2')
    sns.stripplot(data=df, x='Es_Conversacional', y='Estrellas', color=".25", alpha=0.5, jitter=True)
    plt.title("Satisfacción según Tipo de Consulta")
    plt.xticks([0, 1], ['Palabras Clave (Keywords)', 'Conversacional ("Show me...")'])
    plt.xlabel("Estilo de la Consulta")
    plt.ylabel("Valoración Media (Estrellas)")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig11_tipo_consulta_vs_satisfaccion.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Nuevas gráficas generadas.")

if __name__ == "__main__":
    generar_nuevas_graficas()
