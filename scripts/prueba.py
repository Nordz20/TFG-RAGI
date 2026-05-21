import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Asegurarnos de que la carpeta de destino existe
os.makedirs("imagenes", exist_ok=True)

# 4 imágenes extraídas de distintos artículos en tu carpeta data
rutas = [
    r"C:\Users\usuario\Downloads\TFG RAGI FINAL\data\2012.00641v2\images\p0001_figure_00001.png",
    r"C:\Users\usuario\Downloads\TFG RAGI FINAL\data\2601.03482v1\images\p0004_figure_00001.png",
    r"C:\Users\usuario\Downloads\TFG RAGI FINAL\data\2601.04170v1\images\p0006_figure_00001.png",
    r"C:\Users\usuario\Downloads\TFG RAGI FINAL\data\2603.22285v1\images\p0002_figure_00001.png"
]

# Títulos orientativos (puedes cambiarlos si ves que la imagen no coincide exactamente con el título)
titulos = [
    '(a) Arquitectura neuronal', 
    '(b) Gráfica temporal de intervenciones', 
    '(c) Gráfica de rendimiento (Líneas)', 
    '(d) Diagrama de flujo de sistema (Pipeline)'
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, ruta, titulo in zip(axes.flatten(), rutas, titulos):
    try:
        img = mpimg.imread(ruta)
        ax.imshow(img)
        ax.set_title(titulo, fontsize=12)
        ax.axis('off') # Quitar los ejes numéricos
    except FileNotFoundError:
        print(f"No se pudo encontrar: {ruta}")

plt.tight_layout()

# Guardar directamente en tu carpeta de imágenes
ruta_salida = os.path.join("imagenes", "dataset_samples.png")
plt.savefig(ruta_salida, dpi=300)
print(f"¡Imagen guardada con éxito en: {ruta_salida}!")