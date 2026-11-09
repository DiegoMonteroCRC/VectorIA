import io
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(ROOT_DIR / ".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
import keras

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.train.lstm import PronosticadorDengue
from src.train.ann import RecomendadorAcciones

app = FastAPI(
    title="VectorIA API",
    description="API REST para Vigilancia Epidemiológica del Dengue en Costa Rica",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELOS = {
    "cnn": None,
    "lstm_pronosticador": None,
    "lstm_modelo": None,
    "ann": None,
    "df_lstm": None,
    "df_cantonal": None,
}


@app.on_event("startup")
def inicializar_modelos():
    ruta_cnn = ROOT_DIR / "models" / "cnn_propia_mosquitos.keras"
    if not ruta_cnn.exists():
        ruta_cnn = ROOT_DIR / "models" / "cnn_propia_mosquitos.h5"
    if ruta_cnn.exists():
        MODELOS["cnn"] = keras.models.load_model(str(ruta_cnn))

    ruta_datos_lstm = ROOT_DIR / "data" / "processed" / "dengue_nacional_procesado.csv"
    if ruta_datos_lstm.exists():
        df_lstm = pd.read_csv(ruta_datos_lstm, parse_dates=["fecha_inicio_semana"])
        MODELOS["df_lstm"] = df_lstm
        pronosticador, mod_lstm = PronosticadorDengue.cargar(df_lstm, str(ROOT_DIR / "models"))
        MODELOS["lstm_pronosticador"] = pronosticador
        MODELOS["lstm_modelo"] = mod_lstm

    try:
        MODELOS["ann"] = RecomendadorAcciones.cargar(str(ROOT_DIR / "models"))
    except Exception as e:
        print(f"Advertencia al cargar ANN: {e}")

    ruta_cantonal = ROOT_DIR / "data" / "raw" / "vectoria_casos_cantonal_anual_real.csv"
    if ruta_cantonal.exists():
        MODELOS["df_cantonal"] = pd.read_csv(ruta_cantonal)


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
    }


@app.get("/health")
def estado_salud():
    return {
        "estado": "operativo",
        "modelos": {
            "cnn": MODELOS["cnn"] is not None,
            "lstm": MODELOS["lstm_modelo"] is not None,
            "ann": MODELOS["ann"] is not None,
            "datos_cantonales": MODELOS["df_cantonal"] is not None,
        },
    }


@app.post("/predict/cnn")
async def predecir_mosquito(archivo: UploadFile = File(...)):
    if MODELOS["cnn"] is None:
        raise HTTPException(status_code=503, detail="Modelo CNN no disponible.")

    try:
        contenido = await archivo.read()
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo de imagen inválido.")

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
        raise HTTPException(status_code=503, detail="Módulo LSTM no cargado.")

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
        raise HTTPException(status_code=503, detail="Módulo LSTM no disponible.")

    try:
        tabla_pred = pronosticador.pronosticar_desde(modelo, solicitud.fecha)
        comp = pronosticador.comparar_con_umbral(modelo, solicitud.fecha)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en pronóstico: {str(e)}")

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
        raise HTTPException(status_code=503, detail="Módulo ANN no disponible.")

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
        raise HTTPException(status_code=500, detail=f"Error en inferencia ANN: {str(e)}")


@app.get("/data/cantones")
def consultar_cantones(anio: Optional[int] = Query(None), provincia: Optional[str] = Query(None)):
    df = MODELOS["df_cantonal"]
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset cantonal no disponible.")

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
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
