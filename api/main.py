import io
import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

# Configuración de rutas del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Configuración de entorno Keras / TensorFlow para evitar interferencias
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(ROOT_DIR / ".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VectorIA-API")

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
import keras

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.train.lstm import PronosticadorDengue
from src.train.ann import RecomendadorAcciones

# Diccionario centralizado de modelos cargados
MODELOS: Dict[str, Any] = {
    "cnn": None,
    "cnn_nombre": None,
    "lstm_pronosticador": None,
    "lstm_modelo": None,
    "ann": None,
    "df_lstm": None,
    "df_cantonal": None,
    "errores": {},
}


def cargar_modelo_cnn() -> None:
    """Carga aislada y resiliente del modelo CNN para identificación entomológica."""
    nombre_elegido = os.environ.get("VECTORIA_CNN_MODEL", "cnn_propia_mosquitos")
    carpeta_modelos = ROOT_DIR / "models"
    
    rutas_candidatas = [
        carpeta_modelos / f"{nombre_elegido}.keras",
        carpeta_modelos / f"{nombre_elegido}.h5",
        carpeta_modelos / "cnn_propia_mosquitos.keras",
        carpeta_modelos / "cnn_propia_mosquitos.h5",
        carpeta_modelos / "cnn_mobilenetv2_mosquitos.keras",
        carpeta_modelos / "cnn_mobilenetv2_mosquitos.h5",
    ]
    
    for ruta in rutas_candidatas:
        if ruta.exists():
            try:
                MODELOS["cnn"] = keras.models.load_model(str(ruta), compile=False)
                MODELOS["cnn_nombre"] = ruta.name
                logger.info(f"Modelo CNN cargado exitosamente desde: {ruta.name}")
                return
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)}"
                logger.warning(f"Error cargando candidato CNN {ruta.name}: {msg}")
                MODELOS["errores"][f"cnn_{ruta.name}"] = msg
        else:
            MODELOS["errores"][f"cnn_no_existe_{ruta.name}"] = "Archivo no encontrado"
                
    logger.error("No fue posible inicializar ningún modelo CNN.")


def cargar_modelo_lstm() -> None:
    """Carga aislada del modelo recurrente LSTM y su dataset histórico."""
    ruta_datos_lstm = ROOT_DIR / "data" / "processed" / "dengue_nacional_procesado.csv"
    carpeta_modelos = ROOT_DIR / "models"
    
    if not ruta_datos_lstm.exists():
        msg = f"Dataset para LSTM no encontrado: {ruta_datos_lstm}"
        logger.error(msg)
        MODELOS["errores"]["lstm"] = msg
        return
        
    try:
        df_lstm = pd.read_csv(ruta_datos_lstm, parse_dates=["fecha_inicio_semana"])
        MODELOS["df_lstm"] = df_lstm
        pronosticador, mod_lstm = PronosticadorDengue.cargar(df_lstm, str(carpeta_modelos))
        MODELOS["lstm_pronosticador"] = pronosticador
        MODELOS["lstm_modelo"] = mod_lstm
        logger.info("Modelo LSTM y serie histórica cargados exitosamente.")
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error cargando módulo LSTM: {msg}")
        MODELOS["errores"]["lstm"] = msg


def cargar_modelo_ann() -> None:
    """Carga aislada de la red densa ANN de recomendación sanitaria."""
    carpeta_modelos = ROOT_DIR / "models"
    try:
        MODELOS["ann"] = RecomendadorAcciones.cargar(str(carpeta_modelos))
        logger.info("Modelo ANN y preprocesadores cargados exitosamente.")
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error cargando módulo ANN: {msg}")
        MODELOS["errores"]["ann"] = msg


def cargar_datos_cantonales() -> None:
    """Carga del dataset de vigilancia cantonal histórica."""
    ruta_cantonal = ROOT_DIR / "data" / "raw" / "vectoria_casos_cantonal_anual_real.csv"
    if ruta_cantonal.exists():
        try:
            MODELOS["df_cantonal"] = pd.read_csv(ruta_cantonal)
            logger.info("Dataset cantonal cargado exitosamente.")
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error cargando dataset cantonal: {msg}")
            MODELOS["errores"]["cantonal"] = msg
    else:
        logger.warning(f"Dataset cantonal no encontrado en: {ruta_cantonal}")
        MODELOS["errores"]["cantonal"] = f"No encontrado: {ruta_cantonal}"


def inicializar_todos_los_modelos():
    """Ejecuta la carga independiente de cada componente."""
    logger.info("Inicializando subsistemas de Machine Learning / Deep Learning...")
    cargar_modelo_cnn()
    cargar_modelo_lstm()
    cargar_modelo_ann()
    cargar_datos_cantonales()
    logger.info("Inicialización completada.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor moderno de ciclo de vida (startup/shutdown) para FastAPI."""
    inicializar_todos_los_modelos()
    yield
    logger.info("Apagando API VectorIA...")


app = FastAPI(
    title="VectorIA API",
    description="API REST para Vigilancia Epidemiológica del Dengue en Costa Rica (CUC)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolicitudLSTM(BaseModel):
    fecha: str = Field(..., example="2024-06-02")


class SolicitudANN(BaseModel):
    casos_pred_4sem: float = Field(..., ge=0, example=950.0)
    casos_umbral_endemico: float = Field(..., gt=0, example=680.0)
    lluvia_14d_mm: float = Field(..., ge=0, example=65.0)
    temp_max: float = Field(..., example=28.5)
    aegypti_detectado: int = Field(..., ge=0, le=1, example=1)


@app.get("/")
def inicio():
    return {
        "proyecto": "VectorIA",
        "descripcion": "API REST de soporte a la vigilancia del Dengue en Costa Rica",
        "institucion": "Colegio Universitario de Cartago (CUC)",
        "documentacion": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
def estado_salud():
    modelos_estado = {
        "cnn": MODELOS["cnn"] is not None,
        "lstm": MODELOS["lstm_modelo"] is not None and MODELOS["lstm_pronosticador"] is not None,
        "ann": MODELOS["ann"] is not None,
        "datos_cantonales": MODELOS["df_cantonal"] is not None,
    }
    
    activos = sum(1 for v in modelos_estado.values() if v)
    total = len(modelos_estado)
    
    if activos == total:
        estado_global = "operativo"
        codigo_http = status.HTTP_200_OK
    elif activos > 0:
        estado_global = "degradado"
        codigo_http = status.HTTP_200_OK
    else:
        estado_global = "no_disponible"
        codigo_http = status.HTTP_503_SERVICE_UNAVAILABLE

    carpeta_m = ROOT_DIR / "models"
    archivos = [f.name for f in carpeta_m.glob("*")] if carpeta_m.exists() else []

    contenido = {
        "estado": estado_global,
        "modelos_activos": f"{activos}/{total}",
        "cnn_modelo_activo": MODELOS.get("cnn_nombre"),
        "modelos": modelos_estado,
        "errores": MODELOS.get("errores", {}),
        "archivos_modelos": archivos,
    }
    return JSONResponse(status_code=codigo_http, content=contenido)


@app.post("/predict/cnn")
def predecir_mosquito(archivo: UploadFile = File(...)):
    """
    Identificación entomológica mediante CNN.
    Se ejecuta como endpoint síncrono para que FastAPI lo delegue al threadpool de workers
    evitando bloquear el bucle de eventos asíncrono.
    """
    if MODELOS["cnn"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo CNN no disponible en el servidor.",
        )

    try:
        contenido = archivo.file.read()
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Archivo de imagen inválido o corrupto.",
        )

    img_resized = imagen.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    predicciones = MODELOS["cnn"].predict(img_batch, verbose=0)[0]
    clases = ["Aedes aegypti", "Aedes albopictus", "Culex"]
    idx_max = int(np.argmax(predicciones))
    especie_predicha = clases[idx_max]
    confianza = float(predicciones[idx_max]) * 100.0

    probabilidades = {
        clase: round(float(prob) * 100.0, 2)
        for clase, prob in zip(clases, predicciones)
    }

    return {
        "especie": especie_predicha,
        "confianza": round(confianza, 2),
        "aegypti_detectado": 1 if especie_predicha == "Aedes aegypti" else 0,
        "es_vector": especie_predicha in ["Aedes aegypti", "Aedes albopictus"],
        "probabilidades": probabilidades,
    }


@app.get("/data/semanas")
def listar_semanas():
    pronosticador = MODELOS["lstm_pronosticador"]
    if pronosticador is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Módulo LSTM no cargado.",
        )

    semanas = pronosticador.semanas_disponibles(incluir_futuro=True)
    return {
        "total": len(semanas),
        "semanas": [pd.to_datetime(s).strftime("%Y-%m-%d") for s in semanas],
    }


@app.post("/predict/lstm")
def pronosticar_casos(solicitud: SolicitudLSTM):
    pronosticador = MODELOS["lstm_pronosticador"]
    modelo = MODELOS["lstm_modelo"]

    if pronosticador is None or modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Módulo LSTM no disponible.",
        )

    try:
        tabla_pred = pronosticador.pronosticar_desde(modelo, solicitud.fecha)
        comp = pronosticador.comparar_con_umbral(modelo, solicitud.fecha)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error en pronóstico: {str(e)}")

    detalle = []
    for fecha_idx, fila in tabla_pred.iterrows():
        item = {
            "fecha": pd.to_datetime(fecha_idx).strftime("%Y-%m-%d"),
            "forecast": int(round(fila["Forecast"])),
        }
        if "Casos reales" in fila and not pd.isna(fila["Casos reales"]):
            item["real"] = int(round(fila["Casos reales"]))
        else:
            item["real"] = None
        detalle.append(item)

    suma_4sem = int(comp["Pronóstico 4 semanas"])
    umbral_4sem = float(comp["Umbral 4 semanas"]) if not np.isnan(comp["Umbral 4 semanas"]) else 0.0
    razon = float(comp["Razón pronóstico / umbral"]) if not np.isnan(comp["Razón pronóstico / umbral"]) else 1.0

    return {
        "fecha_solicitada": solicitud.fecha,
        "casos_pred_4sem": suma_4sem,
        "casos_umbral_endemico": round(umbral_4sem, 1),
        "razon_umbral": round(razon, 2),
        "situacion": comp["Situación"],
        "tipo": comp["Tipo"],
        "detalle_semanal": detalle,
    }


@app.post("/predict/ann")
def recomendar_accion(solicitud: SolicitudANN):
    ann = MODELOS["ann"]
    if ann is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Módulo ANN no disponible.",
        )

    try:
        resultado = ann.predecir(
            casos_pred_4sem=solicitud.casos_pred_4sem,
            casos_umbral_endemico=solicitud.casos_umbral_endemico,
            lluvia_14d_mm=solicitud.lluvia_14d_mm,
            temp_max=solicitud.temp_max,
            aegypti_detectado=solicitud.aegypti_detectado,
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error en inferencia ANN: {str(e)}")


@app.get("/data/cantones")
def consultar_cantones(anio: Optional[int] = Query(None), provincia: Optional[str] = Query(None)):
    df = MODELOS["df_cantonal"]
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset cantonal no disponible.",
        )

    df_res = df.copy()
    if anio is not None:
        df_res = df_res[df_res["anio"] == anio]
    if provincia is not None and provincia not in ["Todas", "Todas las provincias"]:
        df_res = df_res[df_res["provincia"] == provincia]

    anios_lista = [int(a) for a in sorted(df["anio"].unique())]
    provs_lista = [str(p) for p in sorted(df["provincia"].unique())]

    datos_lista = [
        {
            "anio": int(r["anio"]),
            "provincia": str(r["provincia"]),
            "canton": str(r["canton"]),
            "casos_dengue": int(r["casos_dengue"]),
        }
        for _, r in df_res.iterrows()
    ]

    return {
        "total_registros": len(datos_lista),
        "anios_disponibles": anios_lista,
        "provincias_disponibles": provs_lista,
        "datos": datos_lista,
    }


if __name__ == "__main__":
    import uvicorn
    host_env = os.environ.get("HOST", "0.0.0.0")
    port_env = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api.main:app", host=host_env, port=port_env, reload=False)
