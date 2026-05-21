import os
import ssl
from PIL import Image
import urllib.request

# Desactivar verificación SSL (para entornos con certificados corporativos)
ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = "imagenes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# URLs de iconos ODS (open-sdg.github.io - repositorio oficial de traducciones ONU)
ods_urls = {
    4: "https://open-sdg.github.io/sdg-translations/assets/img/goals/es/4.png",
    9: "https://open-sdg.github.io/sdg-translations/assets/img/goals/es/9.png",
    10: "https://open-sdg.github.io/sdg-translations/assets/img/goals/es/10.png",
    12: "https://open-sdg.github.io/sdg-translations/assets/img/goals/es/12.png",
}

nombres = {
    4: "ODS 4 - Educación de Calidad",
    9: "ODS 9 - Industria, Innovación e Infraestructura",
    10: "ODS 10 - Reducción de las Desigualdades",
    12: "ODS 12 - Producción y Consumo Responsables",
}

temp_dir = "temp_ods"
os.makedirs(temp_dir, exist_ok=True)

imagenes = []
for ods, url in ods_urls.items():
    ruta_local = os.path.join(temp_dir, f"ods{ods}.png")
    print(f"Descargando {nombres[ods]}...")
    try:
        urllib.request.urlretrieve(url, ruta_local)
        img = Image.open(ruta_local).convert("RGBA")
        # Redimensionar todas al mismo tamaño (400x400)
        img = img.resize((400, 400), Image.LANCZOS)
        imagenes.append(img)
        print(f"  ✓ Descargado")
    except Exception as e:
        print(f"  ✗ Error: {e}")

if len(imagenes) == 4:
    # Crear collage 2x2
    ancho_total = 400 * 2
    alto_total = 400 * 2
    collage = Image.new("RGBA", (ancho_total, alto_total), (255, 255, 255, 255))

    # Posiciones: (0,0), (1,0), (0,1), (1,1)
    posiciones = [(0, 0), (400, 0), (0, 400), (400, 400)]
    for img, (x, y) in zip(imagenes, posiciones):
        collage.paste(img, (x, y), img)

    collage.save(os.path.join(OUTPUT_DIR, "ods_involucrados.png"))
    print(f"\n✅ Collage guardado en 'imagenes/ods_involucrados.png'")
else:
    print(f"\n❌ Solo se descargaron {len(imagenes)} de 4 imágenes. No se generó el collage.")

# Limpiar temporales
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)
