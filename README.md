# RAGI: Retrieval-Augmented Generation for Images

RAGI es un sistema de búsqueda semántica multimodal diseñado para la recuperación y enriquecimiento de figuras en artículos científicos. Utiliza IA Generativa y modelos de Embeddings para entender el contenido visual y técnico de las imágenes extraídas de PDFs.

## Requisitos Previos

Para desplegar este sistema, asegúrate de tener instalados en tu servidor/máquina local:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Git
- Python 3.10+ (solo necesario para ejecutar la extracción de PDFs localmente)

---

## ⚙️ Configuración Previa: Red de Ollama

El proyecto está diseñado para comunicarse con un contenedor de **Ollama** que proporciona los modelos de lenguaje locales (LLMs) y de embeddings. En el archivo `docker-compose.yml`, se especifica una red externa llamada `ollama_default`. 

Si vas a desplegar este repositorio desde cero en tu propia máquina (fuera del servidor original de la UPM), **necesitarás crear esta red manualmente** antes de levantar los contenedores de Docker, o de lo contrario el despliegue fallará:

```bash
docker network create ollama_default
```

*(Opcional)* Si no tienes un servidor de Ollama corriendo en esa red, puedes levantar uno básico con:
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama --network ollama_default ollama/ollama
```
Asegúrate de descargar en Ollama los modelos necesarios (por ejemplo, `ollama run llava` o el modelo de embeddings configurado) para que la ingesta y el backend funcionen correctamente.

---

## 🚀 Despliegue de la Arquitectura

El proyecto está completamente contenedorizado para facilitar su puesta en marcha. Sigue estos pasos para desplegar el sistema:

### 1. Clonar el repositorio
Descarga el código fuente en tu máquina:
```bash
git clone https://github.com/Nordz20/TFG-RAGI.git
cd TFG-RAGI
```

### 2. Levantar los servicios
Para construir las imágenes de los contenedores (Frontend y Backend) y levantar la base de datos (Elasticsearch), ejecuta el siguiente comando. 

*Nota: Es fundamental incluir el flag `--build` para garantizar que los cambios en el código y las variables de entorno se apliquen correctamente en los contenedores.*

```bash
docker compose up -d --build
```

### 3. Verificar el estado
Comprueba que los tres contenedores (`ragi_frontend`, `ragi_backend`, `ragi_elasticsearch`) están corriendo correctamente:
```bash
docker ps
```

### 4. Acceso al Sistema
El sistema expone los siguientes puertos configurados en el `docker-compose.yml`:
- **Frontend (Interfaz de Usuario):** http://localhost:3080/ragi
  *(Servidor UPM: http://wiig.dia.fi.upm.es/ragi)*
- **Backend (API y Documentación Swagger):** http://localhost:3081/ragi/docs
- **Exportación de Valoraciones (CSV):** http://localhost:3081/ragi/export_ratings

*(Nota: El sistema está configurado con rutas dinámicas (`/ragi`) para funcionar tras un proxy inverso en el servidor de producción).*

---

## 📄 Proceso de Ingesta y Carga de PDFs/Imágenes

Para añadir nuevos documentos científicos al sistema y que aparezcan en el buscador, debes ejecutar un pipeline de procesamiento offline (scripts en Python) que extrae las imágenes, genera metadatos multimodales usando IA, y los indexa en Elasticsearch.

### Preparación del Entorno (Local)
Es altamente recomendable crear un entorno virtual de Python, ya que la extracción utiliza librerías de Visión por Computador e IA.
```bash
python -m venv venv
# Activar en Linux/Mac:
source venv/bin/activate
# Activar en Windows:
venv\Scripts\activate

# Instalar las librerías necesarias:
pip install PyMuPDF Pillow numpy doclayout-yolo huggingface-hub requests elasticsearch
```

### Paso a Paso para la Ingesta:

1. **Depósito de Documentos:** 
   Coloca todos los archivos PDF que desees procesar dentro de la carpeta raíz `pdfs/`.

2. **Extracción de Imágenes:**
   Ejecuta el script de extracción. Este script analizará los PDFs, recortará las figuras y tablas, y creará las estructuras de carpetas y metadatos base dentro del directorio `data/`.
   ```bash
   python scripts/01_extraccion_imagenes.py
   ```

3. **Generación de Descripciones y Embeddings (Requiere Ollama):**
   A continuación, ejecuta secuencialmente los scripts de procesamiento LLM. Estos scripts toman las imágenes extraídas en `data/` y se comunican con tu instancia de Ollama para generar descripciones ricas, captions mejorados y posibles preguntas (Q&A) de cada imagen.
   ```bash
   python scripts/02_descripciones_llm.py
   python scripts/03_descripciones_llm_caption.py
   python scripts/04_preguntas.py
   ```

4. **Indexación en Elasticsearch:**
   Asegúrate de que tus contenedores de Docker están levantados (`docker compose up -d`). Una vez que Elasticsearch esté corriendo en el puerto 9200 (a nivel interno/backend), ejecuta el script de indexación para cargar todos los JSON procesados y sus vectores (embeddings) en la base de datos.
   ```bash
   python scripts/05_indexacion.py
   ```

¡Listo! Si recargas el Frontend en `http://localhost:3080/ragi`, las nuevas imágenes y documentos ya serán buscables mediante texto natural gracias al motor RAG.

---

## 📝 Notas Técnicas
- **Persistencia de Datos:** Las imágenes extraídas se leen mediante un volumen montado directamente desde la carpeta local `./data` hacia el directorio `/data` del backend. La base de datos vectorial utiliza un volumen interno de Docker (`es_data`) para no perder la indexación al reiniciar los contenedores.
- **Enrutamiento:** FastAPI utiliza el parámetro `root_path="/ragi"` y React utiliza `"homepage": "/ragi"` para garantizar la integridad de las rutas absolutas tras el proxy inverso del servidor. La detección del origen (`window.location.origin`) se realiza de manera dinámica en el cliente.
