"""
app.py — Dashboard Interactivo INVICTUS (Water2Fraud)
=====================================================
Lanzar con:
    streamlit run dashboard/app.py
"""

import sys
import os
import pandas as pd
import streamlit as st

# ── Configuración de ruta para importar src/ ──────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DatasetKeys
from dashboard.data_loader import (
    load_dataframe, load_geodataframe, filter_dataframe,
    aggregate_by_barrio, FEATURES_DISPONIBLES, BARRIOS_ALICANTE,
)
from dashboard.components.map_view       import render_map
from dashboard.components.anomaly_panel  import render_anomaly_panel
from dashboard.components.whatif_simulator import render_whatif
from dashboard.components.llm_report     import render_llm_report


# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="INVICTUS — Dashboard Fraude Turístico",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Global ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fondo oscuro app */
.stApp {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #0d1b2a 100%);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(13,27,42,0.95) !important;
    border-right: 1px solid rgba(76,201,240,0.2);
}

/* Métricas */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(76,201,240,0.15);
    border-radius: 10px;
    padding: 12px;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 4px;
}
[data-baseweb="tab"] {
    border-radius: 8px;
}

/* Headers */
h1, h2, h3 { color: #4cc9f0; }
h4 { color: #a8dadc; }

/* Divider */
hr { border-color: rgba(76,201,240,0.2) !important; }

/* Botón primario */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #1b4965, #4cc9f0);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(76,201,240,0.4);
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #1b4965, #4cc9f0) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1b2a; }
::-webkit-scrollbar-thumb { background: #4cc9f0; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS (cacheada)
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("⏳ Cargando datos del pipeline Water2Fraud..."):
    df_full = load_dataframe()
    gdf     = load_geodataframe()

is_mock = not (
    hasattr(df_full, "_source") or  # marca real (no aplica)
    (os.path.exists(os.path.join(os.path.dirname(__file__), "..",
     "internal", "processed", "AMAEM-2022-2024_not_scaled.csv")))
)


# ════════════════════════════════════════════════════════════════════════════
# 3. ESTADO DE SESIÓN
# ════════════════════════════════════════════════════════════════════════════
if "barrio_seleccionado" not in st.session_state:
    st.session_state.barrio_seleccionado = None


# ════════════════════════════════════════════════════════════════════════════
# 4. SIDEBAR — CONTROLES GLOBALES
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / Título
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:40px;">🌊</div>
        <div style="font-size:20px; font-weight:700; color:#4cc9f0;">INVICTUS</div>
        <div style="font-size:11px; color:#888; margin-top:2px;">Water2Fraud · Detector Turístico</div>
    </div>
    """, unsafe_allow_html=True)

    if is_mock:
        st.warning("⚠️ Usando **datos sintéticos** de demostración. Ejecuta el pipeline para datos reales.", icon="🔬")

    st.markdown("---")
    st.markdown("#### 🗓 Filtro Temporal")

    fechas_disponibles = sorted(df_full[DatasetKeys.FECHA].dt.to_period("M").unique())
    fecha_min = df_full[DatasetKeys.FECHA].min().to_pydatetime()
    fecha_max = df_full[DatasetKeys.FECHA].max().to_pydatetime()

    col_f1, col_f2 = st.columns(2)
    fecha_inicio = col_f1.date_input("Desde", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
    fecha_fin    = col_f2.date_input("Hasta", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

    st.markdown("#### 🏙 Filtro por Barrio")
    barrios_lista = ["Todos los barrios"] + sorted(df_full[DatasetKeys.BARRIO].unique().tolist())
    barrio_filtro = st.selectbox("Barrio / Contrato", barrios_lista, key="sidebar_barrio")

    st.markdown("#### 🌡 Feature del Mapa de Calor")
    feature_label = st.radio(
        "Variable a visualizar:",
        list(FEATURES_DISPONIBLES.keys()),
        index=2, # Ratio Consumo/Contrato por defecto
        key="feature_radio",
    )
    feature_col = FEATURES_DISPONIBLES[feature_label]

    st.markdown("#### 🚽 Filtro por Uso")
    usos_disponibles = ["Todos los usos"] + sorted(df_full[DatasetKeys.USO].unique().tolist())
    # Recomendamos DOMESTICO para que coincida con la IA
    uso_filtro = st.selectbox("Tipo de uso", usos_disponibles, 
                              index=usos_disponibles.index("DOMESTICO") if "DOMESTICO" in usos_disponibles else 0)


    st.markdown("---")
    # Selector de barrio por clic en mapa
    if st.session_state.barrio_seleccionado:
        st.markdown(
            f"""<div style="background:rgba(76,201,240,0.1); border:1px solid #4cc9f0;
            border-radius:8px; padding:10px; text-align:center; font-size:13px;">
            📍 <b>Barrio activo:</b><br>
            <span style="color:#4cc9f0; font-size:15px;">{st.session_state.barrio_seleccionado}</span>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("✖ Deseleccionar barrio", key="deselect_btn"):
            st.session_state.barrio_seleccionado = None
            st.rerun()

    st.markdown("---")
    st.caption("v1.0 · Proyecto INVICTUS · Alicante 2024")


# ════════════════════════════════════════════════════════════════════════════
# 5. FILTRADO DE DATOS
# ════════════════════════════════════════════════════════════════════════════
df_filtered = filter_dataframe(df_full, fecha_inicio, fecha_fin, barrio_filtro, uso_filtro)
df_barrio_agg = aggregate_by_barrio(df_filtered)


# ════════════════════════════════════════════════════════════════════════════
# 6. ENCABEZADO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 10px 0 20px;">
    <h1 style="margin:0; font-size:28px; color:#4cc9f0;">
        🌊 INVICTUS — Dashboard de Detección de Fraude Turístico
    </h1>
    <p style="color:#888; font-size:13px; margin-top:4px;">
        Detección de viviendas turísticas ilegales en Alicante · LSTM-Autoencoder + Reglas Físicas
    </p>
</div>
""", unsafe_allow_html=True)

# KPIs globales rápidos
total_contratos = int(df_filtered[DatasetKeys.NUM_CONTRATOS].sum()) if DatasetKeys.NUM_CONTRATOS in df_filtered.columns else 0
total_alertas   = int(df_filtered[DatasetKeys.ALERTA_TURISTICA_ILEGAL].sum()) if DatasetKeys.ALERTA_TURISTICA_ILEGAL in df_filtered.columns else 0
total_consumo   = df_filtered[DatasetKeys.CONSUMO].sum() if DatasetKeys.CONSUMO in df_filtered.columns else 0
num_barrios     = df_filtered[DatasetKeys.BARRIO].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("🏘 Barrios analizados",    num_barrios)
k2.metric("📋 Contratos procesados",  f"{total_contratos:,}")
k3.metric("💧 Consumo total (m³)",    f"{total_consumo:,.0f}")
k4.metric("🚨 Alertas activas",       total_alertas,
          delta="⚠️" if total_alertas > 10 else None,
          delta_color="inverse" if total_alertas > 10 else "off")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# 7. TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════════════════
tab_mapa, tab_whatif, tab_informe = st.tabs([
    "🗺 Mapa de Calor Interactivo",
    "🔬 Simulador What-if",
    "🤖 Informe LLM",
])


# ─── TAB 1: MAPA ────────────────────────────────────────────────────────────
with tab_mapa:
    col_mapa, col_panel = st.columns([4, 1.4], gap="medium")

    with col_mapa:
        st.markdown(f"#### 🌡 Mapa de Calor — *{feature_label}*")
        st.caption(
            f"Periodo: **{fecha_inicio}** → **{fecha_fin}** · "
            f"{'Todos los barrios' if barrio_filtro == 'Todos los barrios' else barrio_filtro}"
        )
        map_output = render_map(df_barrio_agg, feature_col, gdf)

        # Detectar clic en el mapa (nombre del barrio desde popup)
        if map_output and map_output.get("last_active_drawing"):
            props = map_output["last_active_drawing"].get("properties", {})
            nombre_click = (
                props.get("barrio_limpio") or
                props.get("DENOMINACI") or
                props.get("popup") or
                props.get("barrio")
            )
            if nombre_click and st.session_state.barrio_seleccionado != nombre_click.upper():
                st.session_state.barrio_seleccionado = nombre_click.upper()
                st.rerun()

    with col_panel:
        if st.session_state.barrio_seleccionado:
            render_anomaly_panel(df_filtered, st.session_state.barrio_seleccionado)
        else:
            st.markdown("""
            <div style="
                height: 550px; display: flex; align-items: center;
                justify-content: center; text-align: center;
                background: rgba(255,255,255,0.03);
                border: 1px dashed rgba(76,201,240,0.3);
                border-radius: 12px; color: #668;
            ">
                <div>
                    <div style="font-size: 40px; margin-bottom: 12px;">📍</div>
                    <div style="font-size: 14px; color: #aaa;">
                        Haz clic en un barrio<br>del mapa para ver<br>el panel de anomalías
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 2: WHAT-IF ─────────────────────────────────────────────────────────
with tab_whatif:
    render_whatif(df_filtered, st.session_state.barrio_seleccionado)


# ─── TAB 3: INFORME LLM ────────────────────────────────────────────────────
with tab_informe:
    render_llm_report(st.session_state.barrio_seleccionado)
