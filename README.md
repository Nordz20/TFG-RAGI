# RAGI: Retrieval-Augmented Generation for Images

RAGI es un sistema de búsqueda semántica multimodal diseñado para la recuperación y enriquecimiento de figuras en artículos científicos. Utiliza IA Generativa y modelos de Embeddings para entender el contenido visual y técnico de las imágenes extraídas de PDFs.

## Requisitos Previos

Para desplegar este sistema, asegúrate de tener instalados en tu servidor/máquina local:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Git

## Despliegue Paso a Paso

El proyecto está completamente contenedorizado para facilitar su puesta en marcha. Sigue estos pasos para desplegar el sistema:

### 1. Clonar el repositorio
Descarga el código fuente en el servidor de destino:
```bash
git clone [https://github.com/Nordz20/TFG-RAGI.git](https://github.com/Nordz20/TFG-RAGI.git)
cd TFG-RAGI
```

### 2. Levantar la arquitectura
Para construir las imágenes de los contenedores (Frontend y Backend) y levantar los servicios (incluyendo Elasticsearch), ejecuta el siguiente comando.

Nota: Es fundamental incluir el flag --build para garantizar que los cambios en el código y las variables de entorno se apliquen correctamente en los contenedores.

```bash
docker compose up -d --build
```

### 3. Verificar el estado
Una vez termine el proceso de construcción, puedes comprobar que los tres contenedores (ragi_frontend, ragi_backend, ragi_elasticsearch) están corriendo con:

```bash
docker ps
```
Acceso al Sistema
El sistema está configurado de forma dinámica para funcionar tanto en local como bajo un subdominio específico.

Frontend (Interfaz de Usuario): - Local: http://localhost:80/ragi (o simplemente http://localhost)

Servidor UPM: http://wiig.dia.fi.upm.es/ragi

Backend (API y Documentación Swagger): - Local: http://localhost:8000/ragi/docs

Exportación de Valoraciones (CSV):

Enlace directo: [URL_BASE]/ragi/export_ratings

Notas Técnicas sobre la Configuración
Persistencia de Datos: Las imágenes extraídas se leen mediante un volumen montado directamente desde ./data hacia el directorio /data del backend. La base de datos vectorial utiliza un volumen de Docker (es_data) para no perder la indexación al reiniciar.

Enrutamiento: FastAPI utiliza el parámetro root_path="/ragi" y React utiliza "homepage": "/ragi" para garantizar la integridad de las rutas absolutas tras el proxy inverso del servidor. La detección del origen (window.location.origin) se realiza de manera dinámica en el cliente.