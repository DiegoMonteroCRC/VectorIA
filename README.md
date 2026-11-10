# VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

Proyecto 2 del Colegio Universitario de Cartago (CUC).

VectorIA es un sistema que combina tres modelos de redes neuronales para apoyar la vigilancia epidemiológica y entomológica del dengue en Costa Rica:

| Módulo | Modelo | Función Principal |
|---|---|---|
| **1. Identificación Entomológica** | CNN (Propia / MobileNetV2) | Clasifica la especie del espécimen (*Aedes aegypti*, *Aedes albopictus* o *Culex*) a partir de imágenes |
| **2. Pronóstico Epidemiológico** | RNN / LSTM | Pronostica la incidencia de casos para las próximas 4 semanas y compara con el canal endémico |
| **3. Recomendación Sanitaria** | ANN (Red Densa) | Determina la acción sanitaria preventiva prioritaria según el pronóstico, clima y presencia vectorial |
| **4. Vigilancia Cantonal** | Análisis Geoespacial | Visualiza la distribución histórica de casos y mapa de calor por cantón y provincia |
| **5. Simulación de Escenarios** | Inferencia Paramétrica | Evaluación de sensibilidad y respuesta ante cambios hipotéticos de clima y vectores |

> **Aviso:** Esta es una herramienta técnica de apoyo a la vigilancia epidemiológica y entomológica. No realiza diagnóstico clínico individual de dengue en personas.

---

## Arquitectura de Despliegue en Producción

Para garantizar disponibilidad 24/7 sin costos y sin depender de computadoras locales encendidas, el proyecto está desacoplado en dos capas:

```
[ Usuario en Internet ]
          │
          ▼ HTTPS
┌───────────────────────────────────────────────┐
│     Frontend: Streamlit Community Cloud       │
│  - app/Home.py (Ligero, sin TensorFlow local) │
│  - Muestras entomológicas en app/assets/      │
└───────────────────────┬───────────────────────┘
                        │ API REST (JSON / HTTPS)
                        ▼
┌───────────────────────────────────────────────┐
│       Backend: FastAPI en Contenedor Docker   │
│  - api/main.py (Inferencia CNN, LSTM, ANN)    │
│  - Alojamiento gratuito 24/7:                 │
│    • Oracle Cloud Always Free (Recomendado)   │
│    • Hugging Face Spaces (Docker Free)        │
│    • Render (Web Service Free)                │
└───────────────────────────────────────────────┘
```

---

## Estructura del Repositorio

```
VectorIA/
├── api/
│   └── main.py                     # API REST FastAPI (Lifespan, endpoints de inferencia)
├── app/
│   ├── Home.py                     # Interfaz web interactiva en Streamlit
│   └── assets/
│       └── samples/                # Muestras ligeras de referencia (Aedes aegypti, albopictus, Culex)
├── data/
│   ├── processed/
│   │   ├── clima_nasa_2025_2026.csv      # Clima para proyecciones de mediano/largo plazo
│   │   └── dengue_nacional_procesado.csv # Serie semanal 2014-2024 para modelo LSTM
│   └── raw/
│       ├── vectoria_casos_cantonal_anual_real.csv # Datos históricos cantonales 2013-2018
│       ├── vectoria_dengue_nacional_real.csv      # Datos brutos de casos nacionales
│       └── vectoria_acciones.csv                  # Matriz de entrenamiento ANN
├── deploy/
│   ├── oracle_cloud.md             # Guía paso a paso para Oracle Cloud Always Free (24/7 permanente)
│   └── render_huggingface.md       # Guía para despliegue en Hugging Face Spaces o Render
├── models/
│   ├── cnn_propia_mosquitos.keras / .h5   # Pesos de la CNN propia
│   ├── cnn_mobilenetv2_mosquitos.keras    # Pesos de CNN MobileNetV2
│   ├── modelo_lstm_casos.h5               # Pesos de la red LSTM
│   ├── modelo_lstm_casos_escaladores.pkl  # Escaladores MinMax del pronosticador
│   ├── modelo_ann_acciones.keras          # Pesos de la red ANN
│   └── modelo_ann_acciones_preprocesadores.pkl # Preprocesadores del recomendador
├── notebooks/                      # Cuadernos Jupyter de entrenamiento y experimentación
├── src/                            # Código modular (data_prep.py, clima_nasa.py y src/train/)
├── Dockerfile                      # Imagen Docker optimizada (Python 3.11 + tensorflow-cpu)
├── docker-compose.yml              # Orquestación de backend para servidores VPS/Oracle Cloud
├── requirements.txt                # Dependencias para Streamlit Community Cloud (Frontend)
├── requirements-frontend.txt       # Dependencias exclusivas de interfaz
└── requirements-backend.txt        # Dependencias exclusivas de API e inferencia
```

---

## Endpoints de la API REST

Cuando la API está activa, la documentación interactiva Swagger se encuentra en `/docs`:

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Información del proyecto e institución |
| `GET` | `/health` | Estado del servicio y de cada subsistema (`cnn`, `lstm`, `ann`, `datos_cantonales`) |
| `POST` | `/predict/cnn` | Clasificación de mosquito por imagen (multipart/form-data) |
| `GET` | `/data/semanas` | Lista de semanas epidemiológicas históricas y futuras disponibles |
| `POST` | `/predict/lstm` | Pronóstico de casos a 4 semanas y comparación con umbral endémico |
| `POST` | `/predict/ann` | Evaluación de variables y recomendación sanitaria con distribución de probabilidad |
| `GET` | `/data/cantones` | Consulta de registros y agregaciones cantonales por año y provincia |

---

## Despliegue Gratuito en la Nube

### 1. Despliegue del Backend (FastAPI)

Elige una de las siguientes opciones gratuitas:

- **Opción A (Recomendada - Permanente 24/7): Oracle Cloud Always Free**  
  Sigue la guía detallada en [`deploy/oracle_cloud.md`](deploy/oracle_cloud.md) para levantar la API en una máquina virtual gratuita de por vida con Docker Compose y HTTPS automático.
- **Opción B: Hugging Face Spaces (Docker Free)**  
  Sigue la guía en [`deploy/render_huggingface.md`](deploy/render_huggingface.md) para desplegar el `Dockerfile` en un Space gratuito con 16 GB de RAM y URL HTTPS incluida.
- **Opción C: Render (Web Service Free)**  
  Conecta el repositorio a Render seleccionando entorno Docker (guía en [`deploy/render_huggingface.md`](deploy/render_huggingface.md)).

### 2. Despliegue del Frontend (Streamlit Community Cloud)

1. Ve a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
2. Haz clic en **New app**.
3. Selecciona tu repositorio: `DiegoMonteroCRC/VectorIA`.
4. Rama: `main` (o la rama de despliegue).
5. Archivo principal: `app/Home.py`.
6. En **Advanced Settings** > **Secrets**, define la URL pública de tu backend:
   ```toml
   API_URL = "https://tu-api-en-oracle-o-render.com"
   ```
7. Haz clic en **Deploy**. ¡Tu aplicación quedará pública en internet de inmediato!

---

## Ejecución Local para Desarrollo

Si deseas probar o desarrollar en tu computadora:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/DiegoMonteroCRC/VectorIA.git
   cd VectorIA
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / Mac:
   source .venv/bin/activate
   ```

3. **Instalar dependencias completas:**
   ```bash
   pip install -r requirements-backend.txt
   pip install -r requirements-frontend.txt
   ```

4. **Levantar el Backend (Terminal 1):**
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Levantar el Frontend (Terminal 2):**
   ```bash
   streamlit run app/Home.py
   ```

---

## Variables de Entorno

| Variable | Ámbito | Descripción | Valor por Defecto |
|---|---|---|---|
| `VECTORIA_API_URL` | Frontend | URL base del backend REST | `http://127.0.0.1:8000` |
| `PORT` | Backend | Puerto de escucha del servidor ASGI | `8000` |
| `HOST` | Backend | Dirección de enlace de red | `0.0.0.0` |
| `VECTORIA_CNN_MODEL` | Backend | Arquitectura CNN a cargar (`cnn_propia_mosquitos` o `cnn_mobilenetv2_mosquitos`) | `cnn_propia_mosquitos` |
| `TF_ENABLE_ONEDNN_OPTS` | Backend | Optimizaciones numéricas oneDNN | `0` |

---

## Resultados de los Modelos

### Módulo 1: CNN (Identificación de Mosquitos)
- **CNN Propia:** Accuracy 0.76, F1-macro 0.77.
- **MobileNetV2 (Transfer Learning):** Accuracy 0.82, F1-macro 0.82.

### Módulo 2: LSTM (Pronóstico de Casos)
Evaluación en el periodo de prueba 2023–2024:
- **Baseline estacional:** RMSE 638.3, MAE 455.7
- **SimpleRNN:** RMSE 231.3, MAE 144.0
- **LSTM (Principal):** RMSE 173.4, MAE 105.8 (Reducción sustancial del error)

### Módulo 3: ANN (Recomendación Sanitaria)
Clasificador de 4 acciones preventivas evaluado con validación cruzada:
- **Accuracy:** 0.94
- **F1 macro:** 0.92

---

## Integrantes

- Mariana Méndez Pérez
- Luis Diego Montero Vargas
- Nadin Rojas López
- Claret Rodríguez Jiménez
