import io
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="VectorIA - Vigilancia Dengue Costa Rica",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .header-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.1rem;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 0.8rem;
    }
    .action-panel {
        padding: 16px;
        border-radius: 6px;
        color: #FFFFFF;
        margin-top: 10px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 8px 16px;
        background-color: #1E293B;
        color: #CBD5E1;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 6px 6px 0 0;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7B68EE !important;
        color: #FFFFFF !important;
        border-color: #7B68EE !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolver_url_api_predeterminada() -> str:
    """Resuelve la dirección del backend consultando variables de entorno y secrets."""
    # 1. Variable de entorno (útil para Docker / local)
    if "VECTORIA_API_URL" in os.environ and os.environ["VECTORIA_API_URL"].strip():
        return os.environ["VECTORIA_API_URL"].strip().rstrip("/")
    # 2. Streamlit Secrets (producción en Streamlit Community Cloud)
    try:
        if "API_URL" in st.secrets and str(st.secrets["API_URL"]).strip():
            return str(st.secrets["API_URL"]).strip().rstrip("/")
        if "general" in st.secrets and "API_URL" in st.secrets["general"]:
            return str(st.secrets["general"]["API_URL"]).strip().rstrip("/")
    except Exception:
        pass
    # 3. Valor predeterminado para desarrollo local
    return "http://127.0.0.1:8000"


def consultar_salud_api(base_url: str) -> Optional[Dict[str, Any]]:
    """Verifica la conectividad y estado de salud de la API REST."""
    if not base_url:
        return None
    try:
        r = requests.get(f"{base_url}/health", timeout=3.5)
        if r.status_code in [200, 207]:
            return r.json()
    except Exception:
        pass
    return None


def api_predecir_cnn(base_url: str, img_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Consulta el endpoint de inferencia CNN de mosquitos."""
    try:
        archivos = {"archivo": (filename, img_bytes, "image/jpeg")}
        r = requests.post(f"{base_url}/predict/cnn", files=archivos, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Error en API CNN ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Error de conexión con la API CNN: {e}")
    return None


def api_obtener_semanas(base_url: str) -> List[str]:
    """Obtiene la lista de semanas epidemiológicas disponibles en el backend."""
    try:
        r = requests.get(f"{base_url}/data/semanas", timeout=6)
        if r.status_code == 200:
            return r.json().get("semanas", [])
    except Exception:
        pass
    return []


def api_predecir_lstm(base_url: str, fecha: str) -> Optional[Dict[str, Any]]:
    """Consulta el pronóstico de casos a 4 semanas mediante el modelo LSTM."""
    try:
        r = requests.post(f"{base_url}/predict/lstm", json={"fecha": fecha}, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Error en API LSTM ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Error de conexión con la API LSTM: {e}")
    return None


def api_predecir_ann(base_url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Consulta el modelo ANN de recomendación de medidas sanitarias."""
    try:
        r = requests.post(f"{base_url}/predict/ann", json=payload, timeout=12)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Error en API ANN ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Error de conexión con la API ANN: {e}")
    return None


def api_obtener_cantones(base_url: str, anio: Optional[int] = None, provincia: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Consulta registros cantonales consolidados en la API."""
    params = {}
    if anio is not None:
        params["anio"] = anio
    if provincia is not None and provincia != "Todas":
        params["provincia"] = provincia
    try:
        r = requests.get(f"{base_url}/data/cantones", params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# Inicialización de estado en sesión
if "api_url" not in st.session_state:
    st.session_state["api_url"] = resolver_url_api_predeterminada()
if "aegypti_detectado" not in st.session_state:
    st.session_state["aegypti_detectado"] = 1
if "especie_detectada" not in st.session_state:
    st.session_state["especie_detectada"] = "Aedes aegypti"
if "confianza_especie" not in st.session_state:
    st.session_state["confianza_especie"] = 100.0
if "casos_pred_4sem" not in st.session_state:
    st.session_state["casos_pred_4sem"] = 1200
if "casos_umbral_endemico" not in st.session_state:
    st.session_state["casos_umbral_endemico"] = 900
if "ratio_casos" not in st.session_state:
    st.session_state["ratio_casos"] = 1.33
if "lluvia_14d_mm" not in st.session_state:
    st.session_state["lluvia_14d_mm"] = 75.0
if "temp_max" not in st.session_state:
    st.session_state["temp_max"] = 28.5

# Barra lateral de configuración y monitoreo
with st.sidebar:
    st.markdown("### 🌐 Conexión al Backend")
    api_url_input = st.text_input(
        "Dirección API REST:",
        value=st.session_state["api_url"],
        help="URL pública HTTPS del backend (ej: https://api.vectoria.cr o http://127.0.0.1:8000 en local)",
    )
    if api_url_input.strip() != st.session_state["api_url"]:
        st.session_state["api_url"] = api_url_input.strip().rstrip("/")
        st.rerun()

    estado_api = consultar_salud_api(st.session_state["api_url"])
    api_online = estado_api is not None

    st.markdown("---")
    st.markdown("### 📊 Estado de Subsistemas")
    if api_online:
        estado_global = estado_api.get("estado", "operativo")
        if estado_global == "operativo":
            st.success("🟢 API REST: En línea (Operativa)")
        else:
            st.warning(f"🟡 API REST: En línea ({estado_global.capitalize()})")

        modelos_estado = estado_api.get("modelos", {})
        cnn_nom = estado_api.get("cnn_modelo_activo", "CNN")
        st.write(f"• **CNN ({cnn_nom}):** {'✅ Activo' if modelos_estado.get('cnn') else '❌ Inactivo'}")
        st.write(f"• **LSTM (Pronóstico):** {'✅ Activo' if modelos_estado.get('lstm') else '❌ Inactivo'}")
        st.write(f"• **ANN (Recomendación):** {'✅ Activo' if modelos_estado.get('ann') else '❌ Inactivo'}")
        st.write(f"• **Datos Cantonales:** {'✅ Activo' if modelos_estado.get('datos_cantonales') else '❌ Inactivo'}")
    else:
        st.error("🔴 API REST: Desconectada")
        st.info(
            "El backend no responde en la dirección configurada. "
            "Si la API está en servicio gratuito (Render/Koyeb), puede tardar ~50 segundos en despertar."
        )
        if st.button("🔄 Reintentar conexión"):
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Variables Epidemiológicas Activas")
    st.write(f"• Vector detectado: **{st.session_state['especie_detectada']}**")
    st.write(f"• Aedes aegypti: **{'Positivo (1)' if st.session_state['aegypti_detectado'] == 1 else 'Negativo (0)'}**")
    st.write(f"• Pronóstico 4 sem: **{st.session_state['casos_pred_4sem']:,} casos**")
    st.write(f"• Umbral endémico: **{st.session_state['casos_umbral_endemico']:,} casos**")
    st.write(f"• Razón de exceso: **{st.session_state['ratio_casos']:.2f}x**")


st.markdown('<div class="header-title">VectorIA: Vigilancia Epidemiológica del Dengue</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Vigilancia Entomológica y Epidemiológica · Costa Rica · CUC</div>', unsafe_allow_html=True)

st.warning(
    "⚠️ Aviso: Herramienta de soporte a la vigilancia epidemiológica y entomológica. "
    "No realiza diagnóstico clínico individual de dengue en personas."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Identificación Entomológica (CNN)",
    "2. Pronóstico Epidemiológico (LSTM)",
    "3. Recomendación Sanitaria (ANN)",
    "4. Vigilancia Cantonal",
    "5. Simulación de Escenarios",
])

# -----------------------------------------------------------------------------
# TAB 1: IDENTIFICACIÓN ENTOMOLÓGICA (CNN)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("#### Identificación de Mosquitos por Imagen (Red Convolucional CNN)")

    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        modo_fuente = st.radio(
            "Origen de la imagen:",
            ["Muestras de referencia", "Subir imagen"],
            horizontal=True,
        )

        img_bytes_envio = None
        nombre_archivo_envio = ""
        imagen_mostrar = None

        # Directorios de muestras (prioriza app/assets/samples/ para producción ligera)
        samples_dir = ROOT_DIR / "app" / "assets" / "samples"
        raw_images_dir = ROOT_DIR / "data" / "raw" / "images"

        ejemplos = {
            "Aedes aegypti (Vector principal)": samples_dir / "aedes_aegypti.jpg",
            "Aedes albopictus (Vector secundario)": samples_dir / "aedes_albopictus.jpg",
            "Culex (No vector de dengue)": samples_dir / "culex.jpg",
        }

        # Fallback local a data/raw/images si no estuvieran en assets
        if not (samples_dir / "aedes_aegypti.jpg").exists() and raw_images_dir.exists():
            ejemplos = {
                "Aedes aegypti (Vector principal)": raw_images_dir / "Aedes aegypti" / "Aedes1.jpg",
                "Aedes albopictus (Vector secundario)": raw_images_dir / "Aedes albopictus" / "480a95d1-0f94-4388-9fc5-c0ab59b5822e.jpg",
                "Culex (No vector de dengue)": raw_images_dir / "Culex" / "0df8b757-6bee-477d-a359-8e4be54f893d.jpg",
            }

        if modo_fuente == "Muestras de referencia":
            sel_esp = st.selectbox("Espécimen de referencia:", list(ejemplos.keys()))
            archivo_muestra = ejemplos[sel_esp]

            if archivo_muestra.exists():
                with open(archivo_muestra, "rb") as f:
                    img_bytes_envio = f.read()
                nombre_archivo_envio = archivo_muestra.name
                imagen_mostrar = Image.open(io.BytesIO(img_bytes_envio))
            else:
                st.info("Muestra no encontrada localmente. Puede subir una foto propia en la otra opción.")
        else:
            subido = st.file_uploader("Seleccione archivo JPG o PNG:", type=["jpg", "jpeg", "png"])
            if subido is not None:
                img_bytes_envio = subido.getvalue()
                nombre_archivo_envio = subido.name
                imagen_mostrar = Image.open(io.BytesIO(img_bytes_envio))

        if imagen_mostrar is not None:
            st.image(imagen_mostrar, caption=f"Muestra: {nombre_archivo_envio}", use_container_width=True)

    with c_col2:
        st.markdown("##### Resultado de Análisis Entomológico")

        if img_bytes_envio is not None:
            if not api_online:
                st.error("⚠️ La API REST no está conectada. Inicie el backend o verifique la URL en la barra lateral para clasificar la imagen.")
            else:
                with st.spinner("Procesando imagen con la Red CNN en la API REST..."):
                    res_cnn = api_predecir_cnn(st.session_state["api_url"], img_bytes_envio, nombre_archivo_envio)

                if res_cnn is not None:
                    esp_res = res_cnn["especie"]
                    conf_res = res_cnn["confianza"]
                    aegypti_flag = res_cnn["aegypti_detectado"]

                    st.session_state["especie_detectada"] = esp_res
                    st.session_state["confianza_especie"] = conf_res
                    st.session_state["aegypti_detectado"] = aegypti_flag

                    if esp_res == "Aedes aegypti":
                        st.error(f"Especie detectada: **{esp_res}** (Certeza: {conf_res:.2f}%)")
                        st.markdown("**Clasificación:** Vector primario activo de Dengue en Costa Rica.")
                    elif esp_res == "Aedes albopictus":
                        st.warning(f"Especie detectada: **{esp_res}** (Certeza: {conf_res:.2f}%)")
                        st.markdown("**Clasificación:** Vector secundario de importancia entomológica.")
                    else:
                        st.success(f"Especie detectada: **{esp_res}** (Certeza: {conf_res:.2f}%)")
                        st.markdown("**Clasificación:** Especie no transmisora de virus del Dengue.")

                    df_probs = pd.DataFrame({
                        "Especie": list(res_cnn["probabilidades"].keys()),
                        "Probabilidad (%)": list(res_cnn["probabilidades"].values()),
                    })
                    fig_cnn = px.bar(
                        df_probs,
                        x="Probabilidad (%)",
                        y="Especie",
                        orientation="h",
                        text=[f"{p:.1f}%" for p in df_probs["Probabilidad (%)"]],
                        color="Especie",
                        color_discrete_map={
                            "Aedes aegypti": "#C53030",
                            "Aedes albopictus": "#DD6B20",
                            "Culex": "#2F855A",
                        },
                    )
                    fig_cnn.update_layout(
                        height=210,
                        showlegend=False,
                        margin=dict(l=0, r=20, t=10, b=10),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_cnn, use_container_width=True)
        else:
            st.info("Seleccione una muestra o cargue una imagen para realizar la identificación.")

# -----------------------------------------------------------------------------
# TAB 2: PRONÓSTICO EPIDEMIOLÓGICO (LSTM)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("#### Proyección de Casos a 4 Semanas vs. Canal Endémico (LSTM)")

    semanas_lista = []
    if api_online:
        semanas_lista = api_obtener_semanas(st.session_state["api_url"])

    if not semanas_lista:
        ruta_p = ROOT_DIR / "data" / "processed" / "dengue_nacional_procesado.csv"
        if ruta_p.exists():
            try:
                df_fechas = pd.read_csv(ruta_p)
                semanas_lista = [str(f) for f in df_fechas["fecha_inicio_semana"].dropna().unique()]
            except Exception:
                pass
        if not semanas_lista:
            semanas_lista = ["2024-06-02", "2024-08-18", "2024-10-27", "2025-07-06"]

    idx_sel = semanas_lista.index("2024-06-02") if "2024-06-02" in semanas_lista else 0
    fecha_sel = st.selectbox("Semana epidemiológica de referencia:", semanas_lista, index=idx_sel)

    if not api_online:
        st.warning("⚠️ La API REST no está conectada. Se requiere conexión con el backend para calcular el pronóstico LSTM.")
    else:
        with st.spinner("Consultando pronóstico en API REST..."):
            res_lstm = api_predecir_lstm(st.session_state["api_url"], fecha_sel)

        if res_lstm:
            casos_pred_val = res_lstm["casos_pred_4sem"]
            umbral_val = res_lstm["casos_umbral_endemico"]
            razon_val = res_lstm["razon_umbral"]
            situacion_val = res_lstm["situacion"]

            st.session_state["casos_pred_4sem"] = casos_pred_val
            st.session_state["casos_umbral_endemico"] = max(int(round(umbral_val)), 1)
            st.session_state["ratio_casos"] = razon_val

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Casos Proyectados (4 sem)", f"{casos_pred_val:,}")
            col_m2.metric("Umbral Endémico (4 sem)", f"{int(round(umbral_val)):,}")
            col_m3.metric("Razón sobre Umbral", f"{razon_val:.2f}x")
            col_m4.metric("Diagnóstico Epidemiológico", situacion_val)

            detalle = res_lstm["detalle_semanal"]
            df_det = pd.DataFrame(detalle)

            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(
                x=df_det["fecha"],
                y=df_det["forecast"],
                mode="lines+markers",
                name="Pronóstico LSTM",
                line=dict(color="#C53030", width=3, dash="dash"),
                marker=dict(size=8),
            ))

            if "real" in df_det.columns and df_det["real"].notna().any():
                fig_l.add_trace(go.Scatter(
                    x=df_det["fecha"],
                    y=df_det["real"],
                    mode="lines+markers",
                    name="Casos Observados",
                    line=dict(color="#2B6CB0", width=2),
                    marker=dict(size=6),
                ))

            if umbral_val > 0:
                umbral_sem = umbral_val / 4.0
                fig_l.add_hline(
                    y=umbral_sem,
                    line_dash="dot",
                    line_color="#DD6B20",
                    annotation_text=f"Umbral semanal ({round(umbral_sem):,} casos)",
                    annotation_position="bottom right",
                )

            fig_l.update_layout(
                title=f"Curva Epidemiológica - Semana: {fecha_sel} ({res_lstm['tipo']})",
                xaxis_title="Fecha de Semana Epidemiológica",
                yaxis_title="Casos Notificados",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_l, use_container_width=True)
            st.dataframe(df_det, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: RECOMENDACIÓN SANITARIA (ANN)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("#### Recomendación Preventiva Integral (Red Densa ANN)")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        in_casos = st.number_input(
            "Casos proyectados a 4 semanas:",
            min_value=0,
            max_value=15000,
            value=int(st.session_state["casos_pred_4sem"]),
            step=50,
        )
        in_umbral = st.number_input(
            "Umbral endémico histórico (4 sem):",
            min_value=1,
            max_value=10000,
            value=int(st.session_state["casos_umbral_endemico"]),
            step=50,
        )

    with col_in2:
        in_lluvia = st.slider(
            "Lluvia acumulada últimos 14 días (mm):",
            0.0, 250.0, float(st.session_state["lluvia_14d_mm"]), 1.0,
        )
        in_temp = st.slider(
            "Temperatura máxima (°C):",
            20.0, 38.0, float(st.session_state["temp_max"]), 0.5,
        )

    with col_in3:
        in_aegypti = st.selectbox(
            "Presencia confirmada de Aedes aegypti:",
            [1, 0],
            index=0 if st.session_state["aegypti_detectado"] == 1 else 1,
            format_func=lambda x: "Positivo (1)" if x == 1 else "Negativo (0)",
        )
        ratio_actual = in_casos / max(in_umbral, 1)
        st.metric("Razón de Exceso", f"{ratio_actual:.2f}x")

    st.markdown("---")

    payload_ann = {
        "casos_pred_4sem": float(in_casos),
        "casos_umbral_endemico": float(in_umbral),
        "lluvia_14d_mm": float(in_lluvia),
        "temp_max": float(in_temp),
        "aegypti_detectado": int(in_aegypti),
    }

    if not api_online:
        st.warning("⚠️ La API REST no está conectada. La evaluación de acciones sanitarias por la red ANN requiere conexión con el backend.")
    else:
        with st.spinner("Evaluando matriz de decisión con la ANN en la API REST..."):
            res_ann = api_predecir_ann(st.session_state["api_url"], payload_ann)

        if res_ann:
            color_accion = res_ann.get("color", "#2B6CB0")
            st.markdown(
                f"""
                <div class="action-panel" style="background-color: {color_accion};">
                    <h3 style="margin: 0; color: #FFFFFF;">Dictamen: {res_ann['nombre']}</h3>
                    <p style="margin: 4px 0 6px 0; font-size: 0.95rem;">
                        <strong>Nivel de Riesgo:</strong> {res_ann['nivel']} &nbsp;|&nbsp; 
                        <strong>Certeza:</strong> {res_ann['confianza']:.1f}%
                    </p>
                    <p style="margin: 0; font-size: 0.92rem;">{res_ann['descripcion']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_j1, col_j2 = st.columns([3, 2])
            with col_j1:
                st.markdown("##### Justificación Técnica")
                st.info(res_ann["justificacion"])

            with col_j2:
                st.markdown("##### Distribución de Probabilidad del Modelo")
                df_ann_p = pd.DataFrame({
                    "Acción": list(res_ann["probabilidades"].keys()),
                    "Probabilidad (%)": list(res_ann["probabilidades"].values()),
                })
                fig_ann = px.bar(
                    df_ann_p,
                    x="Probabilidad (%)",
                    y="Acción",
                    orientation="h",
                    text=[f"{p:.1f}%" for p in df_ann_p["Probabilidad (%)"]],
                )
                fig_ann.update_layout(
                    height=180,
                    margin=dict(l=0, r=20, t=10, b=10),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_ann, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: VIGILANCIA CANTONAL
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("#### Vigilancia Geoespacial y Análisis Cantonal")

    datos_cantones = None
    if api_online:
        datos_cantones = api_obtener_cantones(st.session_state["api_url"])

    df_base_cantones = None
    anios_disp = ["Todos"]
    prov_disp = ["Todas"]

    if datos_cantones and datos_cantones.get("datos"):
        df_base_cantones = pd.DataFrame(datos_cantones["datos"])
        anios_disp = ["Todos"] + [str(a) for a in datos_cantones.get("anios_disponibles", [])]
        prov_disp = ["Todas"] + [str(p) for p in datos_cantones.get("provincias_disponibles", [])]
    else:
        ruta_local_canton = ROOT_DIR / "data" / "raw" / "vectoria_casos_cantonal_anual_real.csv"
        if ruta_local_canton.exists():
            df_base_cantones = pd.read_csv(ruta_local_canton)
            anios_disp = ["Todos"] + [str(a) for a in sorted(df_base_cantones["anio"].unique())]
            prov_disp = ["Todas"] + [str(p) for p in sorted(df_base_cantones["provincia"].unique())]

    if df_base_cantones is None or df_base_cantones.empty:
        st.error("No se encontraron registros cantonales disponibles.")
    else:
        col_cf1, col_cf2 = st.columns(2)
        with col_cf1:
            anio_f = st.selectbox("Año epidemiológico:", anios_disp, key="sel_anio_cant")
        with col_cf2:
            prov_f = st.selectbox("Provincia:", prov_disp, key="sel_prov_cant")

        df_c = df_base_cantones.copy()
        if anio_f != "Todos":
            df_c = df_c[df_c["anio"] == int(anio_f)]
        if prov_f != "Todas":
            df_c = df_c[df_c["provincia"] == prov_f]

        c_agrup = (
            df_c.groupby(["provincia", "canton"])["casos_dengue"]
            .sum()
            .reset_index()
            .sort_values(by="casos_dengue", ascending=False)
        )

        total_casos_c = int(c_agrup["casos_dengue"].sum())
        top_c = c_agrup.iloc[0] if not c_agrup.empty else None

        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Casos Registrados en Filtro", f"{total_casos_c:,}")
        col_k2.metric("Total Cantones Afectados", len(c_agrup))
        if top_c is not None:
            col_k3.metric("Mayor Incidencia", f"{top_c['canton']} ({top_c['provincia']})", f"{int(top_c['casos_dengue']):,} casos")

        col_cg1, col_cg2 = st.columns([3, 2])
        with col_cg1:
            st.markdown("##### Ranking de Cantones Prioritarios (Top 15)")
            fig_top = px.bar(
                c_agrup.head(15),
                x="casos_dengue",
                y="canton",
                orientation="h",
                color="provincia",
                labels={"casos_dengue": "Casos", "canton": "Cantón"},
                text="casos_dengue",
            )
            fig_top.update_layout(
                yaxis=dict(autorange="reversed"),
                height=360,
                margin=dict(l=0, r=20, t=10, b=10),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_cg2:
            st.markdown("##### Distribución Provincial")
            p_agrup = df_c.groupby("provincia")["casos_dengue"].sum().reset_index()
            fig_p = px.pie(p_agrup, values="casos_dengue", names="provincia", hole=0.35)
            fig_p.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("##### Mapa de Calor de Casos por Cantón y Año")
        df_mapa = df_base_cantones.copy()
        if prov_f != "Todas":
            df_mapa = df_mapa[df_mapa["provincia"] == prov_f]
        top_cantones = df_mapa.groupby("canton")["casos_dengue"].sum().nlargest(20).index
        tabla_mapa = (
            df_mapa[df_mapa["canton"].isin(top_cantones)]
            .pivot_table(index="canton", columns="anio", values="casos_dengue", aggfunc="sum", fill_value=0)
            .loc[top_cantones]
        )
        fig_mapa = px.imshow(
            tabla_mapa,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            text_auto=True,
            labels={"x": "Año", "y": "Cantón", "color": "Casos"},
        )
        fig_mapa.update_xaxes(type="category")
        fig_mapa.update_layout(
            height=600,
            margin=dict(l=0, r=20, t=10, b=10),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mapa, use_container_width=True)
        st.caption("Los 20 cantones con más casos entre 2013 y 2018 en la provincia elegida. Permite comparar la incidencia relativa entre regiones.")

# -----------------------------------------------------------------------------
# TAB 5: SIMULACIÓN DE ESCENARIOS
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("#### Análisis de Sensibilidad y Simulación de Escenarios")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s_casos = st.slider("Casos proyectados a 4 semanas:", 100, 5000, 1600, 50)
        s_umbral = st.slider("Umbral endémico estacional:", 100, 2500, 800, 50)
        s_aegypti = st.radio("Detección de Aedes aegypti (CNN):", [1, 0], format_func=lambda x: "Presente (1)" if x == 1 else "Ausente (0)", horizontal=True)

    with col_s2:
        s_lluvia = st.slider("Precipitación acumulada 14 días (mm):", 0.0, 250.0, 95.0, 5.0)
        s_temp = st.slider("Temperatura máxima pronosticada (°C):", 20.0, 36.0, 29.0, 0.5)

    payload_sim = {
        "casos_pred_4sem": float(s_casos),
        "casos_umbral_endemico": float(s_umbral),
        "lluvia_14d_mm": float(s_lluvia),
        "temp_max": float(s_temp),
        "aegypti_detectado": int(s_aegypti),
    }

    if not api_online:
        st.warning("⚠️ La API REST no está conectada. Conecte el backend para simular escenarios epidemiológicos.")
    else:
        with st.spinner("Evaluando simulación con la API REST..."):
            res_sim = api_predecir_ann(st.session_state["api_url"], payload_sim)

        if res_sim:
            color_sim = res_sim.get("color", "#2B6CB0")
            st.markdown(
                f"""
                <div class="action-panel" style="background-color: {color_sim};">
                    <h3 style="margin: 0; color: #FFFFFF;">Acción Recomendada: {res_sim['nombre']}</h3>
                    <p style="margin: 4px 0 6px 0; font-size: 0.95rem;">
                        <strong>Nivel de Prioridad:</strong> {res_sim['nivel']} &nbsp;|&nbsp; 
                        <strong>Certeza:</strong> {res_sim['confianza']:.1f}%
                    </p>
                    <p style="margin: 0; font-size: 0.92rem;">{res_sim['justificacion']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            df_s_p = pd.DataFrame({
                "Acción": list(res_sim["probabilidades"].keys()),
                "Probabilidad (%)": list(res_sim["probabilidades"].values()),
            })
            fig_s = px.bar(
                df_s_p,
                x="Probabilidad (%)",
                y="Acción",
                orientation="h",
                text=[f"{p:.1f}%" for p in df_s_p["Probabilidad (%)"]],
            )
            fig_s.update_layout(
                height=160,
                margin=dict(l=0, r=20, t=10, b=10),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_s, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #718096; font-size: 0.82rem;">
        VectorIA · Proyecto 2 · Inteligencia Artificial · Colegio Universitario de Cartago (CUC)
    </div>
    """,
    unsafe_allow_html=True,
)
