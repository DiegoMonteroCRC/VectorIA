# VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

Proyecto 2 del Colegio Universitario de Cartago (CUC).

VectorIA es un sistema que combina tres modelos de redes neuronales para apoyar la vigilancia del dengue en Costa Rica:

| Módulo | Modelo | Qué hace |
|---|---|---|
| 1 | CNN | Identifica la especie del mosquito en una foto (*Aedes aegypti*, *Aedes albopictus* o *Culex*) |
| 2 | RNN / LSTM | Pronostica los casos de dengue de las próximas 4 semanas usando datos del clima |
| 3 | ANN | Recomienda la acción preventiva según el pronóstico, el clima y la especie detectada |

Una app en Streamlit une los tres modelos y muestra un mapa de calor de casos por cantón.

> **Aviso:** esta es una herramienta de apoyo a la vigilancia. No diagnostica dengue en personas.

## Estructura del proyecto

```
VectorIA/
├── data/
│   ├── raw/          ← datos originales del profe (CSV)
│   └── processed/    ← datos procesados por nuestro código
├── notebooks/        ← 01_EDA, 02_CNN_Mosquitos, 03_LSTM_Casos, 04_ANN_Acciones
├── src/              ← código fuente (data_prep.py, clima_nasa.py y train/)
├── models/           ← modelos entrenados (.keras)
├── api/              ← endpoints REST (opcional)
├── app/              ← app de Streamlit
└── requirements.txt  ← librerías necesarias
```

## Instalación

1. Clonar el repositorio:

```
git clone https://github.com/Mari27MP/VectorIA.git
cd VectorIA
```

2. Crear y activar un ambiente virtual (Windows):

```
python -m venv .venv
.venv\Scripts\activate
```

3. Instalar las librerías:

```
pip install -r requirements.txt
```

## Datos

Los datos están en `data/raw/`:

| Archivo | Tipo | Uso |
|---|---|---|
| `vectoria_dengue_nacional_real.csv` | Real | Módulo 2: 574 semanas (2014-2024) de casos y clima |
| `vectoria_acciones.csv` | Mixto | Módulo 3: base real con etiquetas generadas por reglas |
| `vectoria_casos_cantonal_anual_real.csv` | Real | Mapa de calor por cantón (2013-2018) |
| `vectoria_dengue_semanal.csv` | Sintético | Bonus: predicción por provincia |

Los casos vienen de OPS/PAHO (vía OpenDengue) y el clima de NASA POWER. Las **imágenes de mosquitos** no están en el repositorio por su tamaño; se descargan de IEEE DataPort, Mendeley, Mosquito Alert y Kaggle. El clima de 2025-2026 se descarga con `src/clima_nasa.py` y queda guardado en `data/processed/`.

## Cómo usar los notebooks

Se corren en este orden:

1. `notebooks/01_EDA.ipynb`: análisis exploratorio de los datos.
2. `notebooks/02_CNN_Mosquitos.ipynb`: identificación de mosquitos.
3. `notebooks/03_LSTM_Casos.ipynb`: pronóstico de casos.
4. `notebooks/04_ANN_Acciones.ipynb`: recomendación de acciones.

## Cómo correr la API y la app

1. En una terminal, levantar la API:

```
uvicorn api.main:app --reload
```

La documentación de la API queda en http://127.0.0.1:8000/docs

2. En otra terminal, abrir la app:

```
streamlit run app/Home.py

## Resultados

### Módulo 1: CNN (identificación de mosquitos)

Comparamos una CNN propia (caja blanca) con MobileNetV2 (transfer learning). Resultados en el conjunto de prueba (889 imágenes):

| Modelo | Accuracy | F1 macro |
|---|---|---|
| CNN propia | 0.76 | 0.77 |
| MobileNetV2 | 0.82 | 0.82 |

MobileNetV2 fue mejor, pero ninguno de los dos llegó a la meta del proyecto (F1 macro de 85 %). La clase más difícil para los dos modelos fue *Culex*.

### Módulo 2: LSTM (pronóstico de casos)

Pronóstico de las próximas 4 semanas con ventanas de 26 semanas, en el periodo de prueba (2023-2024):

| Modelo | RMSE | MAE |
|---|---|---|
| Baseline estacional | 638.3 | 455.7 |
| SimpleRNN | 231.3 | 144.0 |
| LSTM | 173.4 | 105.8 |

El LSTM fue el mejor y tuvo bastante menos error que el baseline y que la SimpleRNN.

### Módulo 3: ANN (recomendación de acción)

Clasificador de 4 clases (monitoreo normal, eliminación de criaderos, fumigación focalizada y alerta sanitaria), evaluado con validación cruzada sobre las 2,060 filas:

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Una red | 0.94 | 0.91 |
| Ensamble de 3 redes | 0.94 | 0.92 |

## App de Streamlit

La app une los tres modelos: el usuario sube una foto, elige una semana y recibe una recomendación, además de un mapa de calor de casos por cantón. **Estado: en construcción.**

## Limitaciones

- Los datos cantonales no suman el total nacional (diferencia de definición de caso en la fuente), así que sirven para comparar cantones, no como cifras absolutas.
- Las etiquetas de la ANN, la detección de *Aedes* y la predicción usada en ese dataset son simuladas. Solo el clima, los casos y el umbral endémico son reales.
- El LSTM no anticipó el brote de 2023 cuando se usó para pronosticar varios meses adelante.


## Integrantes

- Mariana Méndez Pérez
- Luis Diego Montero Vargas
- Nadin Rojas López
- Claret Rodríguez Jiménez
