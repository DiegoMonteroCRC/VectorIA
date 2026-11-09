import os
from pathlib import Path
import joblib
import numpy as np

if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = os.path.abspath(".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras


class RecomendadorAcciones:
    ACCIONES_INFO = {
        "alerta_sanitaria": {
            "nombre": "Alerta Sanitaria",
            "color": "#D32F2F",
            "nivel": "Crítico",
            "descripcion": "Activar comités de emergencia cantonales, movilización de recursos hospitalarios y comunicación masiva a la población.",
        },
        "fumigacion_focalizada": {
            "nombre": "Fumigación Focalizada",
            "color": "#E65100",
            "nivel": "Alto",
            "descripcion": "Nebulización espacial y termonebulización en manzanas prioritarias con presencia confirmada de Aedes aegypti y casos por encima del canal endémico.",
        },
        "eliminacion_criaderos": {
            "nombre": "Eliminación de Criaderos",
            "color": "#F57C00",
            "nivel": "Medio",
            "descripcion": "Jornadas comunales de recolección de recipientes con agua estancada y aplicación de biolarvicidas previo a intensificación de lluvias.",
        },
        "monitoreo_normal": {
            "nombre": "Monitoreo Normal",
            "color": "#2E7D32",
            "nivel": "Bajo",
            "descripcion": "Vigilancia epidemiológica pasiva habitual, control biológico de rutina y actualización semanal de registros.",
        },
    }

    def __init__(self, modelo, scaler, label_encoder, features):
        self.modelo = modelo
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.features = features

    @classmethod
    def cargar(cls, ruta_modelos="models"):
        carpeta = Path(ruta_modelos)
        ruta_modelo = carpeta / "modelo_ann_acciones.keras"
        ruta_prep = carpeta / "modelo_ann_acciones_preprocesadores.pkl"

        if not ruta_modelo.exists():
            ruta_zip = carpeta / "modelo_ann_acciones.keras.zip"
            if ruta_zip.exists():
                import shutil
                shutil.copy(ruta_zip, ruta_modelo)

        if not ruta_modelo.exists():
            raise FileNotFoundError(f"No se encontró el modelo ANN en {ruta_modelo}")

        if not ruta_prep.exists():
            raise FileNotFoundError(f"No se encontraron los preprocesadores en {ruta_prep}")

        modelo = keras.models.load_model(str(ruta_modelo))
        prep = joblib.load(str(ruta_prep))

        return cls(
            modelo=modelo,
            scaler=prep["scaler"],
            label_encoder=prep["label_encoder"],
            features=prep["features"],
        )

    def predecir(self, casos_pred_4sem, casos_umbral_endemico, lluvia_14d_mm, temp_max, aegypti_detectado):
        casos_pred_4sem = float(casos_pred_4sem)
        casos_umbral_endemico = float(max(casos_umbral_endemico, 1.0))
        ratio_casos = casos_pred_4sem / casos_umbral_endemico
        lluvia_14d_mm = float(lluvia_14d_mm)
        temp_max = float(temp_max)
        aegypti_detectado = int(aegypti_detectado)

        fila = np.array([[casos_pred_4sem, casos_umbral_endemico, ratio_casos, lluvia_14d_mm, temp_max, aegypti_detectado]])
        fila_escalada = self.scaler.transform(fila)

        pred_probs = self.modelo.predict(fila_escalada, verbose=0)[0]
        idx_max = int(np.argmax(pred_probs))
        accion_clave = self.label_encoder.classes_[idx_max]

        probabilidades = {
            clase: float(prob) * 100
            for clase, prob in zip(self.label_encoder.classes_, pred_probs)
        }

        justificacion = self._generar_justificacion(
            accion_clave=accion_clave,
            casos_pred=casos_pred_4sem,
            umbral=casos_umbral_endemico,
            ratio=ratio_casos,
            lluvia=lluvia_14d_mm,
            temp=temp_max,
            aegypti=aegypti_detectado,
        )

        meta = self.ACCIONES_INFO.get(accion_clave, {
            "nombre": accion_clave.replace("_", " ").title(),
            "color": "#1976D2",
            "nivel": "Informativo",
            "descripcion": "Acción preventiva recomendada por el modelo.",
        })

        return {
            "accion": accion_clave,
            "nombre": meta["nombre"],
            "color": meta["color"],
            "nivel": meta["nivel"],
            "descripcion": meta["descripcion"],
            "confianza": probabilidades[accion_clave],
            "probabilidades": probabilidades,
            "justificacion": justificacion,
            "variables": {
                "casos_pred_4sem": round(casos_pred_4sem),
                "casos_umbral_endemico": round(casos_umbral_endemico),
                "ratio_casos": round(ratio_casos, 2),
                "lluvia_14d_mm": round(lluvia_14d_mm, 1),
                "temp_max": round(temp_max, 1),
                "aegypti_detectado": bool(aegypti_detectado),
            },
        }

    def _generar_justificacion(self, accion_clave, casos_pred, umbral, ratio, lluvia, temp, aegypti):
        puntos = []
        if ratio >= 2.0:
            puntos.append(f"La proyección de casos ({round(casos_pred):,}) supera en más del doble el umbral endémico ({round(umbral):,}), lo que representa una situación de brote inminente o epidemia.")
        elif ratio >= 1.0:
            puntos.append(f"La proyección de casos ({round(casos_pred):,}) está por encima del umbral endémico estacional ({round(umbral):,}), indicando zona de alarma.")
        else:
            puntos.append(f"La proyección de casos ({round(casos_pred):,}) se mantiene dentro o por debajo del umbral endémico ({round(umbral):,}).")

        if aegypti == 1:
            puntos.append("Se detectó presencia activa del vector transmisor (Aedes aegypti) mediante clasificación visual por CNN.")
        else:
            puntos.append("No se detectó Aedes aegypti en las muestras recientes.")

        if lluvia > 80.0 and temp > 27.0:
            puntos.append(f"Las condiciones bioclimáticas son altamente favorables para la reproducción (lluvia reciente: {lluvia:.1f} mm, temp. máx: {temp:.1f} °C).")
        elif lluvia > 50.0:
            puntos.append(f"Precipitaciones moderadas ({lluvia:.1f} mm) que requieren atención sobre recipientes descubiertos.")

        return " ".join(puntos)
