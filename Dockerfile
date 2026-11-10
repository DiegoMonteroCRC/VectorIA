# Dockerfile para VectorIA API (Backend FastAPI + TensorFlow CPU)
FROM python:3.11-slim

# Evitar prompts interactivos y optimizar ejecución de Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    KERAS_HOME=/app/.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    HOST=0.0.0.0 \
    PORT=8000

# Directorio de trabajo
WORKDIR /app

# Instalar utilidades mínimas del sistema para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements-backend.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-backend.txt

# Copiar configuración de Keras
COPY .keras/ /app/.keras/

# Copiar modelos entrenados y artefactos
COPY models/ /app/models/

# Copiar datasets necesarios para inferencia y módulos
COPY data/processed/ /app/data/processed/
COPY data/raw/vectoria_casos_cantonal_anual_real.csv /app/data/raw/vectoria_casos_cantonal_anual_real.csv

# Copiar código fuente
COPY src/ /app/src/
COPY api/ /app/api/

# Exponer puerto
EXPOSE 8000

# Verificación de salud del contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Comando de inicio del servidor ASGI
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
