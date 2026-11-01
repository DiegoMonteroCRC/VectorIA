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
            pd.DataFrame: 2,060 filas (412 semanas x 5 escenarios) con las
                features y la columna objetivo 'recomendacion'.
        """
        df = pd.read_csv(self.carpeta_raw / self.ARCHIVO_ACCIONES)
        df["fecha_inicio_semana"] = pd.to_datetime(df["fecha_inicio_semana"])
        return df

    def cargar_cantonal(self):
        """
        Carga los casos anuales por cantón (mapa de calor del dashboard).

        Nota: la suma de cantones no coincide con el total nacional, por lo
        que estos datos se deben usar como proporciones, no como cifras
        absolutas.

        Returns:
            pd.DataFrame: 422 filas (71 cantones, 2013-2018).
        """
        return pd.read_csv(self.carpeta_raw / self.ARCHIVO_CANTONAL)

    # ------------------------------------------------------------------
    # Transformaciones
    # ------------------------------------------------------------------
    def agregar_logaritmos(self, df):
        """
        Agrega versiones en logaritmo de los casos y de la lluvia.

        Los casos y la lluvia están muy sesgados por los brotes y las
        tormentas extremas; el logaritmo reduce ese sesgo y mejora la
        relación con las demás variables.

        Args:
            df (pd.DataFrame): datos con 'casos_dengue' y 'lluvia_mm'.

        Returns:
            pd.DataFrame: copia de df con las columnas nuevas 'log_casos'
                (log de casos) y 'log_lluvia' (log de 1 + lluvia).
        """
        df = df.copy()
        df["log_casos"] = np.log(df["casos_dengue"])
        df["log_lluvia"] = np.log1p(df["lluvia_mm"])
        return df

    def agregar_rezagos(self, df, rezagos):
        """
        Agrega columnas con valores de semanas anteriores (features con rezago).

        El clima influye en los casos varias semanas después, por eso el
        modelo necesita ver los valores pasados.

        Args:
            df (pd.DataFrame): datos ordenados por fecha.
            rezagos (dict): {columna: [rezagos en semanas]}.
                Ejemplo: {"temp_min_c": [6, 8], "log_lluvia": [3]}

        Returns:
            pd.DataFrame: copia de df con una columna nueva por cada rezago,
                con el nombre '<columna>_lag<k>' (por ejemplo, temp_min_c_lag6).
                Las primeras filas quedan vacías (NaN) porque no tienen
                semanas anteriores.
        """
        df = df.copy()
        for columna, lista in rezagos.items():
            for k in lista:
                df[f"{columna}_lag{k}"] = df[columna].shift(k)
        return df

    def agregar_baseline_estacional(self, df):
        """
        Agrega la predicción del baseline estacional.

        El baseline predice que cada semana tendrá los mismos casos que la
        misma semana del año anterior (52 semanas antes). Es la referencia
        que el LSTM debe superar.

        Args:
            df (pd.DataFrame): datos con 'casos_dengue', ordenados por fecha.

        Returns:
            pd.DataFrame: copia de df con la columna 'baseline_estacional'.
                Las primeras 52 semanas quedan vacías (NaN).
        """
        df = df.copy()
        df["baseline_estacional"] = df["casos_dengue"].shift(52)
        return df

    # ------------------------------------------------------------------
    # División y guardado
    # ------------------------------------------------------------------
    def dividir_temporal(self, df, fecha_corte="2023-01-01"):
        """
        Divide los datos en entrenamiento y prueba según una fecha.

        En series de tiempo no se puede dividir al azar, porque el modelo
        vería datos del futuro. Todo lo anterior a la fecha de corte es
        entrenamiento y lo posterior es prueba.

        Args:
            df (pd.DataFrame): datos con 'fecha_inicio_semana'.
            fecha_corte (str): fecha en formato 'AAAA-MM-DD'.

        Returns:
            tuple: (train, test), dos DataFrames.
        """
        corte = pd.Timestamp(fecha_corte)
        train = df[df["fecha_inicio_semana"] < corte]
        test = df[df["fecha_inicio_semana"] >= corte]
        return train, test

    def guardar_procesado(self, df, nombre_archivo):
        """
        Guarda un DataFrame como CSV en data/processed/.

        Args:
            df (pd.DataFrame): datos a guardar.
            nombre_archivo (str): nombre del archivo, por ejemplo
                'dengue_nacional_procesado.csv'.

        Returns:
            Path: ruta completa del archivo guardado.
        """
        self.carpeta_processed.mkdir(parents=True, exist_ok=True)
        ruta = self.carpeta_processed / nombre_archivo
        df.to_csv(ruta, index=False)
        return ruta


class AnalizadorSerie:
    """
    Cálculos de análisis exploratorio sobre la serie semanal de casos.

    Atributos:
        df (pd.DataFrame): serie semanal ordenada por fecha.
    """

    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): serie semanal (por ejemplo, la que devuelve
                PreparadorDatos.cargar_nacional()).
        """
        self.df = df

    def correlacion_rezagos(self, objetivo, variables, max_rezago=12):
        """
        Calcula la correlación entre el objetivo y cada variable con rezago.

        Sirve para encontrar con cuántas semanas de rezago cada variable
        del clima se relaciona más con los casos.

        Args:
            objetivo (str): columna objetivo, por ejemplo 'log_casos'.
            variables (list): columnas a comparar, por ejemplo
                ['temp_min_c', 'log_lluvia'].
            max_rezago (int): rezago máximo en semanas (por defecto 12).

        Returns:
            pd.DataFrame: una fila por rezago (0..max_rezago) y una columna
                por variable, con la correlación de Pearson.
        """
        tabla = pd.DataFrame(
            {v: [self.df[objetivo].corr(self.df[v].shift(k)) for k in range(max_rezago + 1)]
             for v in variables}
        )
        tabla.index.name = "rezago_semanas"
        return tabla

    def efecto_semana_epi(self, columna="casos_dengue", vecinos=2):
        """
        Detecta semanas del año con subregistro o sobrerregistro de casos.

        Compara cada semana contra el promedio de las semanas vecinas
        (k semanas antes y k después). Una razón cercana a 1 es normal;
        menor a 1 indica menos casos de lo esperado (posible subregistro,
        por ejemplo en vacaciones de fin de año) y mayor a 1 indica casos
        acumulados.

        Args:
            columna (str): columna de casos a analizar.
            vecinos (int): cuántas semanas antes y después usar (por defecto 2).

        Returns:
            pd.Series: mediana de la razón para cada semana epidemiológica.
        """
        s = self.df[columna]
        esperado = (s.shift(vecinos) + s.shift(-vecinos)) / 2
        razon = s / esperado
        return razon.groupby(self.df["semana_epi"]).median()

    @staticmethod
    def errores(real, prediccion):
        """
        Calcula RMSE y MAE entre valores reales y predichos.

        Las filas donde la predicción está vacía (NaN) se ignoran.

        Args:
            real (pd.Series): valores reales.
            prediccion (pd.Series): valores predichos.

        Returns:
            dict: {'RMSE': float, 'MAE': float}, redondeados a 1 decimal.
        """
        mascara = prediccion.notna()
        error = real[mascara] - prediccion[mascara]
        rmse = float(np.sqrt((error ** 2).mean()))
        mae = float(error.abs().mean())
        return {"RMSE": round(rmse, 1), "MAE": round(mae, 1)}
