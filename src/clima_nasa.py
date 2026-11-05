"""
clima_nasa.py - Descarga del clima real de NASA POWER para VectorIA.

Proyecto 2 - Colegio Universitario de Cartago (CUC)
VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

El profesor recomendó usar el clima real de NASA POWER para los escenarios de
2025-2026, en lugar del clima promedio de cada semana. Este módulo arma el
clima semanal nacional con el mismo método del dataset del proyecto
(LEEME_datos.md):

    - NASA POWER diario: lluvia corregida (PRECTOTCORR), temperatura máxima y
      mínima a 2 m (T2M_MAX, T2M_MIN) y humedad relativa (RH2M).
    - En las 7 cabeceras de provincia, promediado y ponderado por población.
    - Agrupado por semana (semanas que empiezan en domingo): lluvia sumada y
      temperaturas y humedad promediadas.

Antes de guardar 2025-2026, se descarga también 2014-2024 y se compara con
vectoria_dengue_nacional_real.csv para comprobar que el cálculo da lo mismo.

Uso (desde la carpeta raíz del proyecto, con el .venv activado):
    python src/clima_nasa.py

Resultado: data/processed/clima_nasa_2025_2026.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests


class ClimaNASAPower:
    """
    Descarga y arma el clima semanal nacional desde la API de NASA POWER.

    Atributos:
        carpeta_raw (Path): carpeta con los CSV originales del proyecto.
        carpeta_processed (Path): carpeta donde se guarda el resultado.
    """

    URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    PARAMETROS = ["PRECTOTCORR", "T2M_MAX", "T2M_MIN", "RH2M"]

    # Nombre de cada parámetro de NASA POWER en el dataset del proyecto
    COLUMNAS = {"PRECTOTCORR": "lluvia_mm", "T2M_MAX": "temp_max_c",
                "T2M_MIN": "temp_min_c", "RH2M": "humedad_rel_pct"}

    # Cabeceras de provincia: (latitud, longitud, población de la provincia).
    # Las poblaciones son las del paquete de datos y suman 5,170,000, la misma
    # población nacional de vectoria_dengue_nacional_real.csv.
    CABECERAS = {
        "San José": (9.9333, -84.0833, 1_660_000),
        "Alajuela": (10.0163, -84.2116, 1_100_000),
        "Cartago": (9.8644, -83.9194, 550_000),
        "Heredia": (9.9981, -84.1170, 520_000),
        "Liberia (Guanacaste)": (10.6346, -85.4377, 390_000),
        "Puntarenas": (9.9763, -84.8384, 500_000),
        "Limón": (9.9907, -83.0359, 450_000),
    }

    def __init__(self):
        raiz = Path(__file__).resolve().parents[1]
        self.carpeta_raw = raiz / "data" / "raw"
        self.carpeta_processed = raiz / "data" / "processed"

    # ------------------------------------------------------------------
    # Descarga
    # ------------------------------------------------------------------
    def descargar_punto(self, latitud, longitud, inicio, fin):
        """
        Descarga el clima diario de un punto desde la API de NASA POWER.

        Args:
            latitud (float), longitud (float): coordenadas del punto.
            inicio (str), fin (str): fechas en formato 'AAAA-MM-DD'.

        Returns:
            pd.DataFrame: una fila por día y una columna por parámetro.
                Los días sin dato (-999 en NASA POWER) quedan como NaN.
        """
        parametros = {
            "parameters": ",".join(self.PARAMETROS),
            "community": "AG",
            "latitude": latitud,
            "longitude": longitud,
            "start": pd.Timestamp(inicio).strftime("%Y%m%d"),
            "end": pd.Timestamp(fin).strftime("%Y%m%d"),
            "format": "JSON",
        }
        # Si la fecha final todavía no está publicada, NASA POWER responde con
        # error: se vuelve a intentar con una semana menos (hasta 8 veces).
        for _ in range(8):
            respuesta = requests.get(self.URL, params=parametros, timeout=120)
            if respuesta.status_code != 422:
                break
            fin_menos = pd.Timestamp(parametros["end"]) - pd.Timedelta(days=7)
            parametros["end"] = fin_menos.strftime("%Y%m%d")
        respuesta.raise_for_status()
        datos = respuesta.json()["properties"]["parameter"]
        diario = pd.DataFrame(datos)
        diario.index = pd.to_datetime(diario.index, format="%Y%m%d")
        return diario.replace(-999, np.nan).astype(float)

    def clima_diario_nacional(self, inicio, fin):
        """
        Clima diario nacional: promedio de las 7 cabeceras ponderado por población.

        Returns:
            pd.DataFrame: una fila por día con lluvia_mm, temp_max_c,
                temp_min_c y humedad_rel_pct.
        """
        poblacion_total = sum(p for _, _, p in self.CABECERAS.values())
        nacional = None
        for nombre, (lat, lon, poblacion) in self.CABECERAS.items():
            print(f"Descargando {nombre}...")
            diario = self.descargar_punto(lat, lon, inicio, fin) * (poblacion / poblacion_total)
            nacional = diario if nacional is None else nacional + diario
        return nacional.rename(columns=self.COLUMNAS)

    @staticmethod
    def clima_semanal(diario):
        """
        Agrupa el clima diario por semana (domingo a sábado).

        La lluvia se suma y las temperaturas y la humedad se promedian. Solo se
        guardan las semanas con los 7 días completos.

        Args:
            diario (pd.DataFrame): clima diario nacional.

        Returns:
            pd.DataFrame: una fila por semana, con 'fecha_inicio_semana'.
        """
        diario = diario.dropna()
        inicio_semana = diario.index - pd.to_timedelta((diario.index.day_of_week + 1) % 7, unit="D")
        grupos = diario.groupby(inicio_semana)
        semanal = pd.DataFrame({
            "lluvia_mm": grupos["lluvia_mm"].sum(),
            "temp_max_c": grupos["temp_max_c"].mean(),
            "temp_min_c": grupos["temp_min_c"].mean(),
            "humedad_rel_pct": grupos["humedad_rel_pct"].mean(),
            "dias": grupos.size(),
        })
        semanal = semanal[semanal["dias"] == 7].drop(columns="dias").round(1)
        semanal.index.name = "fecha_inicio_semana"
        return semanal.reset_index()

    # ------------------------------------------------------------------
    # Comprobación y guardado
    # ------------------------------------------------------------------
    def comparar_con_dataset(self, semanal):
        """
        Compara el clima calculado con el del dataset del proyecto (2014-2024).

        Returns:
            pd.DataFrame: por variable, la correlación, la diferencia promedio
                (sesgo) y la diferencia absoluta promedio.
        """
        real = pd.read_csv(self.carpeta_raw / "vectoria_dengue_nacional_real.csv",
                           parse_dates=["fecha_inicio_semana"])
        unidos = real.merge(semanal, on="fecha_inicio_semana", suffixes=("_dataset", "_nasa"))
        filas = {}
        for columna in self.COLUMNAS.values():
            a, b = unidos[f"{columna}_dataset"], unidos[f"{columna}_nasa"]
            filas[columna] = {
                "semanas": len(unidos),
                "correlación": round(a.corr(b), 3),
                "diferencia promedio": round((b - a).mean(), 2),
                "diferencia absoluta promedio": round((b - a).abs().mean(), 2),
            }
        return pd.DataFrame(filas).T

    def generar(self, inicio_validacion="2013-12-29", fin=None):
        """
        Descarga todo, compara con el dataset y guarda las semanas de 2025 en adelante.

        Args:
            inicio_validacion (str): primera fecha a descargar (inicio del dataset).
            fin (str): última fecha a descargar. Si es None, se usa hoy menos
                5 días (NASA POWER publica los datos con unos días de atraso).

        Returns:
            tuple: (clima semanal de 2025 en adelante, tabla de comparación).
        """
        fin = pd.Timestamp.today().normalize() - pd.Timedelta(days=5) if fin is None else fin
        diario = self.clima_diario_nacional(inicio_validacion, fin)
        semanal = self.clima_semanal(diario)

        comparacion = self.comparar_con_dataset(semanal)
        futuro = semanal[semanal["fecha_inicio_semana"] >= "2024-12-29"]

        self.carpeta_processed.mkdir(parents=True, exist_ok=True)
        ruta = self.carpeta_processed / "clima_nasa_2025_2026.csv"
        futuro.to_csv(ruta, index=False)
        return futuro, comparacion, ruta


if __name__ == "__main__":
    clima = ClimaNASAPower()
    futuro, comparacion, ruta = clima.generar()

    print("\n=== Comparación con el clima del dataset (2014-2024) ===")
    print(comparacion.to_string())
    print(f"\nSemanas de 2025-2026 descargadas: {len(futuro)}")
    if len(futuro):
        print("Desde", futuro["fecha_inicio_semana"].min().date(),
              "hasta", futuro["fecha_inicio_semana"].max().date())
    print("Guardado en:", ruta)
