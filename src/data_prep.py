"""
data_prep.py - Preparación de datos del proyecto VectorIA.

Proyecto 2 - Colegio Universitario de Cartago (CUC)
VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

Este módulo centraliza la carga, limpieza y transformación de los datos del
proyecto, para que los notebooks (EDA, LSTM, ANN) y la app de Streamlit usen
siempre el mismo código y no lo repitan.

Clases:
    PreparadorDatos: carga los CSV y aplica transformaciones (logaritmos,
        rezagos, baseline estacional, división train/test).
    AnalizadorSerie: cálculos de análisis exploratorio sobre la serie
        semanal de casos (correlaciones con rezago, subregistro, errores).

Ejemplo de uso (desde un notebook en la carpeta notebooks/):
    import sys
    sys.path.append("..")
    from src.data_prep import PreparadorDatos, AnalizadorSerie

    preparador = PreparadorDatos()
    df = preparador.cargar_nacional()
    df = preparador.agregar_logaritmos(df)
    train, test = preparador.dividir_temporal(df, "2023-01-01")
"""

from pathlib import Path

import numpy as np
import pandas as pd


class PreparadorDatos:
    """
    Carga y prepara los datos de dengue y clima de VectorIA.

    Las rutas se calculan a partir de la ubicación de este archivo, así que
    la clase funciona igual si se usa desde notebooks/, app/ o la raíz.

    Atributos:
        carpeta_raw (Path): carpeta con los CSV originales (data/raw).
        carpeta_processed (Path): carpeta donde se guardan los datos
            procesados (data/processed).
    """

    # Ruta base del proyecto (la carpeta que contiene src/, data/, notebooks/...)
    RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

    # Nombres de los archivos del paquete VectorIA_datos
    ARCHIVO_NACIONAL = "vectoria_dengue_nacional_real.csv"
    ARCHIVO_ACCIONES = "vectoria_acciones.csv"
    ARCHIVO_CANTONAL = "vectoria_casos_cantonal_anual_real.csv"

    # Variables de clima disponibles en la serie nacional
    VARIABLES_CLIMA = ["lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]

    # Columnas que en el CSV nacional tienen un solo valor y no aportan información
    COLUMNAS_CONSTANTES = ["provincia", "region_climatica", "poblacion"]

    def __init__(self):
        """Inicializa las rutas de las carpetas de datos."""
        self.carpeta_raw = self.RAIZ_PROYECTO / "data" / "raw"
        self.carpeta_processed = self.RAIZ_PROYECTO / "data" / "processed"

    # ------------------------------------------------------------------
    # Carga de archivos
    # ------------------------------------------------------------------
    def cargar_nacional(self):
        """
        Carga la serie semanal nacional de casos y clima (Módulo 2).

        Convierte la fecha a datetime, elimina las columnas constantes
        (provincia, región y población) y agrega el mes del año.

        Returns:
            pd.DataFrame: 574 semanas (2014-2024) con casos y clima.
        """
        df = pd.read_csv(self.carpeta_raw / self.ARCHIVO_NACIONAL)
        df["fecha_inicio_semana"] = pd.to_datetime(df["fecha_inicio_semana"])
        df = df.drop(columns=self.COLUMNAS_CONSTANTES)
        df["mes"] = df["fecha_inicio_semana"].dt.month
        return df

    def cargar_acciones(self):
        """
        Carga el dataset de recomendaciones del Módulo 3 (ANN).

        Returns: