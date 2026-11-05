"""
lstm.py - Módulo 2 de VectorIA: pronóstico de casos de dengue con redes recurrentes.

Proyecto 2 - Colegio Universitario de Cartago (CUC)
VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

Contiene la clase PronosticadorDengue, que hace los mismos pasos del notebook
03_LSTM_Casos.ipynb (y de los ejercicios RNN-Forecast vistos en clase), pero
organizados en métodos para poder reutilizarlos desde la app de Streamlit:

    1. Variables de entrada y variable objetivo.
    2. Train / validación / test (split temporal).
    3. Escalado con MinMaxScaler (ajustado solo con train).
    4. Ventanas con TimeseriesGenerator (26 semanas -> 4 semanas).
    5. Modelos: SimpleRNN (base) y LSTM (principal).
    6. Entrenamiento con EarlyStopping.
    7. Predicción, desescalado y métricas (RMSE, MAE).
    8. Guardado y carga del modelo.

Ejemplo de uso desde la app (con el modelo ya entrenado en el notebook):
    from src.train.lstm import PronosticadorDengue

    pronosticador, modelo = PronosticadorDengue.cargar(df, "models")
    pronosticador.pronosticar_desde(modelo, "2023-07-01")  # semana elegida por el usuario
    pronosticador.pronosticar_futuro(modelo)               # después del último dato
    pronosticador.pronosticar_largo_plazo(modelo, clima_futuro=pronosticador.clima_futuro)  # 2025-2026
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class PronosticadorDengue:
    """
    Pronóstico de los casos de dengue de las próximas semanas con SimpleRNN o LSTM.

    Atributos:
        df (pd.DataFrame): serie semanal con las variables de entrada y el objetivo.
        columnas (list): variables de entrada de la red.
        columnas_objetivo (list): columnas con el cambio de log(casos) a predecir.
        longitud (int): semanas de historia por ventana.
        semanas_pred (int): semanas que se pronostican.
        tamaño_batch (int): ejemplos por lote.
        semilla (int): semilla aleatoria para poder repetir los resultados.
        val_ind (int): fila donde empieza validación.
        test_ind (int): fila donde empieza test.
        scaler (MinMaxScaler): escalador de las variables de entrada.
        scaler_y (MinMaxScaler): escalador del objetivo.
    """

    # Variables de entrada: casos, clima y fecha (elegidas en el EDA)
    COLUMNAS = ["log_casos", "temp_min_c", "log_lluvia", "humedad_rel_pct",
                "temp_max_c", "semana_sin", "semana_cos"]

    def __init__(self, df, columnas=None, longitud=26, semanas_pred=4,
                 tamaño_batch=16, semilla=42):
        """
        Args:
            df (pd.DataFrame): serie semanal con 'fecha_inicio_semana', 'semana_epi',
                'casos_dengue', 'log_casos' y el clima
                (data/processed/dengue_nacional_procesado.csv).
            columnas (list): variables de entrada. Si es None se usa COLUMNAS.
            longitud (int): semanas de historia por ventana (26).
            semanas_pred (int): semanas a pronosticar (4).
            tamaño_batch (int): tamaño de lote (16).
            semilla (int): semilla aleatoria (42).
        """
        self.columnas = columnas or list(self.COLUMNAS)
        self.longitud = longitud
        self.semanas_pred = semanas_pred
        self.tamaño_batch = tamaño_batch
        self.semilla = semilla
        self.columnas_objetivo = [f"cambio_{k + 1}" for k in range(semanas_pred)]
        self.df = self.preparar_variables(df.reset_index(drop=True), semanas_pred)
        self.val_ind = None
        self.test_ind = None
        self.scaler = None
        self.scaler_y = None
        self.clima_futuro = None   # clima real de las semanas futuras (opcional)
        self._largo_plazo = None   # escenario de largo plazo ya calculado

    # ------------------------------------------------------------------
    # 1. Variables
    # ------------------------------------------------------------------
    @staticmethod
    def preparar_variables(df, semanas_pred=4):
        """
        Agrega las variables de fecha y la variable objetivo.

        - semana_sin y semana_cos: la semana del año como un ciclo (la semana 52
          queda cerca de la semana 1).
        - cambio_k: log(casos) de la semana k (contando desde la siguiente) menos
          log(casos) de la última semana conocida. 0 = igual, +0.69 = el doble,
          -0.69 = la mitad.

        Args:
            df (pd.DataFrame): datos con 'semana_epi' y 'log_casos'.
            semanas_pred (int): cuántas columnas cambio_k crear.

        Returns:
            pd.DataFrame: copia de df con las columnas nuevas.
        """
        df = df.copy()
        df["semana_sin"] = np.sin(2 * np.pi * df["semana_epi"] / 52)
        df["semana_cos"] = np.cos(2 * np.pi * df["semana_epi"] / 52)
        for k in range(semanas_pred):
            df[f"cambio_{k + 1}"] = df["log_casos"].shift(-k) - df["log_casos"].shift(1)
        return df

    # ------------------------------------------------------------------
    # 2 y 3. Split y escalado
    # ------------------------------------------------------------------
    def dividir(self, tamaño_test=104, tamaño_val=52):
        """
        Split temporal: las últimas semanas son test y las anteriores, validación.

        Args:
            tamaño_test (int): semanas de test (104 = 2023-2024).
            tamaño_val (int): semanas de validación (52 = 2022).

        Returns:
            tuple: (train, val, test) como DataFrames.
        """
        self.test_ind = len(self.df) - tamaño_test
        self.val_ind = self.test_ind - tamaño_val
        return (self.df.iloc[:self.val_ind],
                self.df.iloc[self.val_ind:self.test_ind],
                self.df.iloc[self.test_ind:])

    def escalar(self, hasta=None):
        """
        Ajusta los MinMaxScaler SOLO con las filas de train y escala toda la serie.

        Args:
            hasta (int): filas [0, hasta) que se usan para ajustar. Si es None,
                solo train (hasta val_ind). Para el modelo final se usa len(df).

        Returns:
            tuple: (datos escalados, objetivo escalado).
        """
        hasta = self.val_ind if hasta is None else hasta
        self.scaler = MinMaxScaler().fit(self.df[self.columnas].iloc[:hasta])
        # Las últimas 3 filas de train tienen cambios que usan semanas de validación: no se usan
        self.scaler_y = MinMaxScaler().fit(
            self.df[self.columnas_objetivo].iloc[:hasta - (self.semanas_pred - 1)])
        self.scaled_data = self.scaler.transform(self.df[self.columnas])
        self.scaled_y = self.scaler_y.transform(self.df[self.columnas_objetivo])
        return self.scaled_data, self.scaled_y

    # ------------------------------------------------------------------
    # 4. Generadores
    # ------------------------------------------------------------------
    def _generador(self, primera_fila, ultima_fila, tamaño_batch):
        """TimeseriesGenerator cuyas predicciones empiezan entre primera_fila y ultima_fila."""
        return TimeseriesGenerator(self.scaled_data, self.scaled_y,
                                   length=self.longitud, batch_size=tamaño_batch,
                                   start_index=primera_fila - self.longitud,
                                   end_index=ultima_fila)

    def crear_generadores(self):
        """
        Crea los generadores de train, validación y test. Las 4 semanas a
        predecir de cada ventana quedan dentro de su propio conjunto.

        Returns:
            tuple: (generador, val_generador, test_generador).
        """
        generador = self._generador(self.longitud, self.val_ind - self.semanas_pred,
                                    self.tamaño_batch)
        val_generador = self._generador(self.val_ind, self.test_ind - self.semanas_pred,
                                        self.tamaño_batch)
        test_generador = self._generador(self.test_ind, len(self.df) - self.semanas_pred, 1)
        return generador, val_generador, test_generador

    def crear_generador_completo(self):
        """Generador con toda la serie, para entrenar el modelo final (forecast)."""
        return self._generador(self.longitud, len(self.df) - self.semanas_pred,
                               self.tamaño_batch)

    # ------------------------------------------------------------------
    # 5 y 6. Modelos y entrenamiento
    # ------------------------------------------------------------------
    def construir_simple_rnn(self):
        """
        Modelo base: SimpleRNN(32) -> Dense(4), optimizador adam y pérdida mse.

        Returns:
            Sequential: modelo compilado.
        """
        tf.keras.utils.set_random_seed(self.semilla)
        model = Sequential()
        model.add(SimpleRNN(32, activation="relu",
                            input_shape=(self.longitud, len(self.columnas))))
        model.add(Dense(self.semanas_pred))
        model.compile(optimizer="adam", loss="mse")
        return model

    def construir_lstm(self):
        """
        Modelo principal: LSTM(64) -> LSTM(32) -> Dropout(0.2) -> Dense(4).

        Returns:
            Sequential: modelo compilado (adam, mse).
        """
        tf.keras.utils.set_random_seed(self.semilla)
        model = Sequential()
        model.add(LSTM(64, activation="relu", return_sequences=True,
                       input_shape=(self.longitud, len(self.columnas))))
        model.add(LSTM(32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(self.semanas_pred))
        model.compile(optimizer="adam", loss="mse")
        return model

    @staticmethod
    def entrenar(model, generador, val_generador=None, epochs=300, paciencia=15):
        """
        Entrena el modelo. Con val_generador usa EarlyStopping y se queda con
        los pesos de la mejor época.

        Args:
            model (Sequential): modelo compilado.
            generador (TimeseriesGenerator): datos de entrenamiento.
            val_generador (TimeseriesGenerator): datos de validación o None.
            epochs (int): máximo de épocas.
            paciencia (int): épocas sin mejorar antes de parar.

        Returns:
            pd.DataFrame: historial de pérdidas (losses).
        """
        callbacks = []
        if val_generador is not None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=paciencia,
                                           restore_best_weights=True))
        model.fit(generador, epochs=epochs, validation_data=val_generador,
                  callbacks=callbacks, verbose=0)
        return pd.DataFrame(model.history.history)

    # ------------------------------------------------------------------
    # 7. Predicción y métricas
    # ------------------------------------------------------------------
    @staticmethod
    def filas_objetivo(generador):
        """Fila de la primera semana pronosticada de cada ventana del generador."""
        return np.arange(generador.start_index, generador.end_index + 1)

    def a_casos(self, predicciones, filas):
        """
        Convierte la salida de la red en número de casos:
        inverse_transform -> + log(casos) de la última semana conocida -> exp.

        Args:
            predicciones (np.ndarray): salida del modelo (ventanas, 4).
            filas (np.ndarray): primera semana pronosticada de cada ventana.

        Returns:
            np.ndarray: casos pronosticados (ventanas, 4).
        """
        cambios = self.scaler_y.inverse_transform(predicciones)
        ultimo_log = self.df["log_casos"].values[filas - 1].reshape(-1, 1)
        return np.exp(cambios + ultimo_log)

    def predecir(self, model, generador):
        """Casos pronosticados para cada ventana del generador (ventanas, 4)."""
        return self.a_casos(model.predict(generador, verbose=0), self.filas_objetivo(generador))

    def casos_reales(self, generador):
        """Casos reales de las 4 semanas de cada ventana del generador (ventanas, 4)."""
        filas = self.filas_objetivo(generador)
        casos = self.df["casos_dengue"].values
        return np.column_stack([casos[filas + k] for k in range(self.semanas_pred)])

    def baseline_estacional(self, generador):
        """Casos de la misma semana del año anterior (52 semanas antes)."""
        filas = self.filas_objetivo(generador)
        casos = self.df["casos_dengue"].values
        return np.column_stack([casos[filas + k - 52] for k in range(self.semanas_pred)])

    @staticmethod
    def evaluar(real, prediccion):
        """
        Métricas del pronóstico.

        Returns:
            dict: RMSE, MAE y precisión (%) = 100 x (1 - suma de errores / suma de casos).
        """
        real, prediccion = np.ravel(real), np.ravel(prediccion)
        return {
            "RMSE": round(float(np.sqrt(mean_squared_error(real, prediccion))), 1),
            "MAE": round(float(mean_absolute_error(real, prediccion)), 1),
            "Precisión (%)": round(float(100 * (1 - np.abs(real - prediccion).sum() / real.sum())), 1),
        }

    # ------------------------------------------------------------------
    # Pronóstico para el dashboard
    # ------------------------------------------------------------------
    @staticmethod
    def semana_epidemiologica(fecha):
        """
        Año y semana epidemiológica de una fecha, con la misma regla de los
        datos: las semanas empiezan en domingo y la semana 1 es la que contiene
        el 3 de enero.

        Args:
            fecha (str o Timestamp): fecha de inicio de la semana.

        Returns:
            tuple: (año epidemiológico, semana epidemiológica).
        """
        fecha = pd.Timestamp(fecha)
        for anio in (fecha.year + 1, fecha.year, fecha.year - 1):
            tres_enero = pd.Timestamp(anio, 1, 3)
            inicio = tres_enero - pd.Timedelta(days=(tres_enero.weekday() + 1) % 7)
            if fecha >= inicio:
                return anio, (fecha - inicio).days // 7 + 1

    @staticmethod
    def cargar_clima_futuro(ruta):
        """
        Lee el clima real de NASA POWER de 2025-2026 que genera src/clima_nasa.py.

        Args:
            ruta (str o Path): ruta de data/processed/clima_nasa_2025_2026.csv.

        Returns:
            pd.DataFrame: clima semanal con 'fecha_inicio_semana', las 4
                variables del clima y 'log_lluvia'. Vacío si el archivo no existe.
        """
        ruta = Path(ruta)
        if not ruta.exists():
            return pd.DataFrame()
        clima = pd.read_csv(ruta, parse_dates=["fecha_inicio_semana"])
        clima["log_lluvia"] = np.log1p(clima["lluvia_mm"])
        return clima

    def pronosticar_largo_plazo(self, model, semanas=104, clima_futuro=None):
        """
        Pronostica más allá de los datos (por ejemplo, 2025 y 2026) repitiendo
        el pronóstico de 4 semanas, como el forecast de varios periodos visto
        en clase: se predicen 4 semanas, se agregan a la ventana y se vuelve
        a predecir.

        Como recomendó el profesor, solo las primeras 4 semanas son un
        pronóstico (Tipo = 'Pronóstico'); las demás son posibles escenarios
        (Tipo = 'Escenario'), porque el error se acumula en cada repetición
        y el modelo no puede anticipar brotes nuevos.

        Clima de las semanas futuras: si se pasa clima_futuro (por ejemplo,
        el clima real de NASA POWER), se usa para las semanas que tenga; para
        las demás se usa el clima promedio de esa semana epidemiológica en
        los datos.

        Args:
            model (Sequential): modelo entrenado con toda la serie.
            semanas (int): semanas a pronosticar (104 = 2 años).
            clima_futuro (pd.DataFrame): clima semanal con 'fecha_inicio_semana',
                'temp_min_c', 'log_lluvia', 'humedad_rel_pct' y 'temp_max_c'.
                Si es None se usa solo el clima promedio.

        Returns:
            pd.DataFrame: índice = fecha de inicio de cada semana, columnas
                'anio', 'semana_epi', 'Forecast', 'Tipo', 'Clima' (de dónde
                salió el clima de esa semana) y 'Año típico' (mediana de los
                casos de esa semana epidemiológica, como referencia).
        """
        clima = ["temp_min_c", "log_lluvia", "humedad_rel_pct", "temp_max_c"]
        clima_promedio = self.df.groupby("semana_epi")[clima].mean()
        clima_real = (pd.DataFrame(columns=clima) if clima_futuro is None or len(clima_futuro) == 0
                      else clima_futuro.set_index("fecha_inicio_semana")[clima])
        historia = self.df.set_index("fecha_inicio_semana")[self.columnas].copy()
        filas = []

        while len(filas) < semanas:
            batch_actual = self.scaler.transform(historia.iloc[-self.longitud:])
            batch_actual = batch_actual.reshape((1, self.longitud, len(self.columnas)))
            cambio = self.scaler_y.inverse_transform(model.predict(batch_actual, verbose=0))[0]
            ultimo_log = historia["log_casos"].iloc[-1]

            for k in range(self.semanas_pred):
                fecha = historia.index[-1] + pd.Timedelta(weeks=1)
                anio, semana = self.semana_epidemiologica(fecha)
                if fecha in clima_real.index:
                    nueva, origen = clima_real.loc[fecha].copy(), "NASA POWER (real)"
                else:
                    nueva = clima_promedio.loc[min(semana, clima_promedio.index.max())].copy()
                    origen = "Promedio de la semana"
                nueva["log_casos"] = ultimo_log + cambio[k]
                nueva["semana_sin"] = np.sin(2 * np.pi * semana / 52)
                nueva["semana_cos"] = np.cos(2 * np.pi * semana / 52)
                historia.loc[fecha] = nueva[self.columnas]
                filas.append({"fecha": fecha, "anio": anio, "semana_epi": semana,
                              "Forecast": round(float(np.exp(nueva["log_casos"]))),
                              "Tipo": "Pronóstico" if len(filas) < self.semanas_pred else "Escenario",
                              "Clima": origen})

        largo = pd.DataFrame(filas).set_index("fecha").iloc[:semanas]
        largo["Año típico"] = largo["semana_epi"].map(self.año_tipico())
        return largo

    def año_tipico(self):
        """
        Mediana de los casos de cada semana epidemiológica en los datos.
        Sirve como referencia de lo que pasa en un año normal.

        Returns:
            pd.Series: casos medianos por semana epidemiológica.
        """
        return self.df.groupby("semana_epi")["casos_dengue"].median()

    def semanas_disponibles(self, incluir_futuro=True):
        """
        Semanas desde las que se puede pronosticar, para el selector del
        dashboard: desde la semana 26 de los datos (necesita 26 semanas de
        historia) hasta 2026 si incluir_futuro es True.

        Returns:
            pd.Series: fechas de inicio de semana disponibles.
        """
        fechas = self.df["fecha_inicio_semana"].iloc[self.longitud - 1:]
        if incluir_futuro:
            futuras = pd.date_range(fechas.iloc[-1] + pd.Timedelta(weeks=1),
                                    periods=104 - self.semanas_pred, freq="7D")
            fechas = pd.concat([fechas, pd.Series(futuras)])
        return fechas.reset_index(drop=True)

    def pronosticar_desde(self, model, fecha):
        """
        Pronostica las 4 semanas siguientes a la semana elegida por el usuario
        (por ejemplo, desde el dashboard).

        - Si la semana está dentro de los datos (2014-2024), el modelo usa las
          26 semanas reales que terminan en ella y se devuelven también los
          casos reales para comparar.
        - Si la semana es de 2025 o 2026, se toma del escenario de largo
          plazo (ver pronosticar_largo_plazo), con el clima real de NASA
          POWER si se cargó en el atributo clima_futuro.

        Args:
            model (Sequential): modelo entrenado.
            fecha (str o Timestamp): cualquier día de la semana elegida
                (por ejemplo, '2023-07-01').

        Returns:
            pd.DataFrame: índice = fecha de inicio de cada semana pronosticada,
                columnas 'Forecast', 'Casos reales' (NaN si no existen) y
                'Año típico' (referencia).
        """
        fecha = pd.Timestamp(fecha)
        fechas_datos = self.df["fecha_inicio_semana"]
        ultima = fechas_datos.iloc[-1]

        # Semana futura (2025-2026): se usa el escenario de largo plazo
        if fecha > ultima + pd.Timedelta(days=6):
            if getattr(self, "_largo_plazo", None) is None:
                self._largo_plazo = self.pronosticar_largo_plazo(
                    model, clima_futuro=getattr(self, "clima_futuro", None))
            largo = self._largo_plazo
            inicio = int(np.searchsorted(largo.index.values, fecha.to_datetime64(), side="right"))
            tabla = largo.iloc[inicio:inicio + self.semanas_pred].copy()
            if len(tabla) < self.semanas_pred:
                raise ValueError(f"Solo se puede pronosticar hasta {largo.index[-1].date()}.")
            tabla["Casos reales"] = np.nan
            return tabla[["Forecast", "Casos reales", "Año típico", "Tipo", "Clima"]]

        semana = int(np.searchsorted(fechas_datos.values, fecha.to_datetime64(), side="right")) - 1
        if semana < self.longitud - 1:
            raise ValueError(f"La semana elegida necesita {self.longitud} semanas de historia. "
                             f"Elija una fecha desde {self.semanas_disponibles().iloc[0].date()}.")

        # Ventana de las 26 semanas que terminan en la semana elegida
        batch_actual = self.scaled_data[semana - self.longitud + 1: semana + 1]
        batch_actual = batch_actual.reshape((1, self.longitud, len(self.columnas)))
        casos = self.a_casos(model.predict(batch_actual, verbose=0), np.array([semana + 1]))[0]

        fechas = pd.date_range(start=fechas_datos.iloc[semana] + pd.Timedelta(weeks=1),
                               periods=self.semanas_pred, freq="7D")
        reales = (self.df.set_index("fecha_inicio_semana")["casos_dengue"]
                  .reindex(fechas).values)
        tipico = [self.año_tipico().get(self.semana_epidemiologica(f)[1]) for f in fechas]
        return pd.DataFrame({"Forecast": casos.round(), "Casos reales": reales,
                             "Año típico": tipico, "Tipo": "Pronóstico",
                             "Clima": "Datos reales"}, index=fechas)

    def pronosticar_futuro(self, model):
        """
        Pronostica las 4 semanas siguientes a la última semana de los datos.

        Returns:
            pd.DataFrame: fecha de inicio de cada semana y casos pronosticados.
        """
        ultima_fecha = self.df["fecha_inicio_semana"].iloc[-1]
        return self.pronosticar_desde(model, ultima_fecha)[["Forecast"]]

    # ------------------------------------------------------------------
    # 8. Guardado y carga
    # ------------------------------------------------------------------
    def guardar(self, model, carpeta, nombre="modelo_lstm_casos"):
        """
        Guarda el modelo (.h5, como en clase) y los escaladores (.pkl), que la
        app necesita para preparar las entradas y desescalar la salida.

        Returns:
            tuple: (ruta del modelo, ruta de los escaladores).
        """
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_modelo = carpeta / f"{nombre}.h5"
        ruta_escaladores = carpeta / f"{nombre}_escaladores.pkl"
        model.save(ruta_modelo)
        joblib.dump({"scaler": self.scaler, "scaler_y": self.scaler_y,
                     "columnas": self.columnas, "longitud": self.longitud,
                     "semanas_pred": self.semanas_pred}, ruta_escaladores)
        return ruta_modelo, ruta_escaladores

    @classmethod
    def cargar(cls, df, carpeta, nombre="modelo_lstm_casos"):
        """
        Carga el modelo guardado para usarlo, por ejemplo, desde la app.

        Args:
            df (pd.DataFrame): serie semanal con al menos las últimas 26 semanas.
            carpeta (str o Path): carpeta con los archivos guardados.
            nombre (str): nombre base de los archivos.

        Returns:
            tuple: (pronosticador, modelo).
        """
        carpeta = Path(carpeta)
        datos = joblib.load(carpeta / f"{nombre}_escaladores.pkl")
        pronosticador = cls(df, columnas=datos["columnas"], longitud=datos["longitud"],
                            semanas_pred=datos["semanas_pred"])
        pronosticador.scaler = datos["scaler"]
        pronosticador.scaler_y = datos["scaler_y"]
        pronosticador.scaled_data = pronosticador.scaler.transform(
            pronosticador.df[pronosticador.columnas])
        # Clima real de 2025-2026 (si ya se generó con src/clima_nasa.py)
        ruta_clima = Path(carpeta).resolve().parent / "data" / "processed" / "clima_nasa_2025_2026.csv"
        pronosticador.clima_futuro = cls.cargar_clima_futuro(ruta_clima)
        modelo = load_model(carpeta / f"{nombre}.h5", compile=False)
        return pronosticador, modelo
