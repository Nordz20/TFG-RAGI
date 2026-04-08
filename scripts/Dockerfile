# Usamos Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Instalamos las librerías
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend.py .

# Exponemos el puerto
EXPOSE 8000

# Arrancamos la API
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]