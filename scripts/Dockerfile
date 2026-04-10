FROM python:3.10-slim

WORKDIR /app

# Copiamos todo el contenido de la carpeta scripts al contenedor
COPY . .

# Instalamos las librerías
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto
EXPOSE 8000

# Comando para arrancar el servidor
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]