"""
app.py — Dashboard Interactivo INVICTUS (Water2Fraud)
=====================================================
Lanzar con:
    streamlit run dashboard/app.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

# ── Patch global del JSON encoder para compatibilidad con Python 3.14 ────────
# Python 3.14 rechaza numpy.int64/float64 en json.dumps (más estricto que 3.12).
# Folium/branca/Jinja2 usan json.dumps internamente sin encoder personalizado,
# por lo que parcheamos JSONEncoder.default UNA vez aquí para toda la sesión.
_orig_json_default = json.JSONEncoder.default

def _numpy_safe_default(self, obj):
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.bool_):     return bool(obj)
    return _orig_json_default(self, obj)

json.JSONEncoder.default = _numpy_safe_default
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuración de pandas y ruta para src/ ──────────────────────────────
pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DatasetKeys
from src.config.paths import Paths
from dashboard.data_loader import (
    load_dataframe, load_geodataframe, filter_dataframe,
    aggregate_by_barrio, FEATURES_DISPONIBLES, BARRIOS_ALICANTE,
)
from dashboard.components.map_view         import render_map
from dashboard.components.whatif_simulator import render_whatif
from dashboard.components.llm_report       import render_llm_report


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA PÁGINA
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="INVICTUS — Dashboard de Anomalías Hídricas",
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
# CARGA DE DATOS (cacheada)
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("Cargando datos del pipeline INVICTUS..."):
    df_full = load_dataframe()
    gdf     = load_geodataframe()

# ════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ════════════════════════════════════════════════════════════════════════════
if "barrio_seleccionado" not in st.session_state:
    st.session_state.barrio_seleccionado = None


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONTROLES GLOBALES
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / Título
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:40px;">🌊</div>
        <div style="font-size:20px; font-weight:700; color:#4cc9f0;">INVICTUS</div>
        <div style="font-size:11px; color:#888; margin-top:2px;">INVICTUS · Atribución Causal</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Filtro Temporal")

    fechas_disponibles = sorted(df_full[DatasetKeys.FECHA].dt.to_period("M").unique())
    meses_es = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun", 
                7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
    opciones_label = [f"{meses_es[f.month]} {f.year}" for f in fechas_disponibles]
    mapeo_fechas = {label: f.to_timestamp() for label, f in zip(opciones_label, fechas_disponibles)}

    # --- Filtro Temporal ---
    if "temp_range" not in st.session_state:
        st.session_state.temp_range = (opciones_label[0], opciones_label[-1])

    st.markdown("##### Selección Rápida")
    c_p1, c_p2 = st.columns(2)
    if c_p1.button("Último Año", width="stretch"):
        st.session_state.temp_range = (opciones_label[-13] if len(opciones_label) >= 13 else opciones_label[0], opciones_label[-1])
        st.rerun()
    if c_p2.button("Todo", width="stretch"):
        st.session_state.temp_range = (opciones_label[0], opciones_label[-1])
        st.rerun()

    # Slider de rango: pasamos 'value' explícitamente para forzar comportamiento de RANGO
    rango_sel = st.select_slider(
        "Rango temporal",
        options=opciones_label,
        value=st.session_state.temp_range
    )
    # Sincronizamos de vuelta al estado manual
    st.session_state.temp_range = rango_sel

    fecha_inicio = mapeo_fechas[rango_sel[0]]
    fecha_fin    = mapeo_fechas[rango_sel[1]] + pd.offsets.MonthEnd(0)


    st.markdown("#### Buscar Barrio")
    todos_barrios = sorted(df_full[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper().unique().tolist())
    query_busqueda = st.text_input(
        "Escribe un barrio...",
        value="",
        placeholder="Ej: Benalua, Juan XXIII...",
        key="buscar_barrio_input",
        label_visibility="collapsed",
    )
    if query_busqueda.strip():
        q = query_busqueda.strip().upper()
        coincidencias = [b for b in todos_barrios if q in b]
        if len(coincidencias) == 1:
            if st.session_state.barrio_seleccionado != coincidencias[0]:
                st.session_state.barrio_seleccionado = coincidencias[0]
                st.rerun()
        elif coincidencias:
            st.markdown("<div style='font-size:12px; color:#888; margin-top:4px;'>Sugerencias:</div>", unsafe_allow_html=True)
            for sug in coincidencias[:6]:
                if st.button(sug, key=f"sug_{sug}", width='stretch'):
                    st.session_state.barrio_seleccionado = sug
                    st.rerun()
        else:
            st.markdown("<div style='font-size:11px; color:#e74c3c;'>Sin coincidencias</div>", unsafe_allow_html=True)

    st.markdown("#### Filtro por Barrio")
    barrios_lista = ["Todos los barrios"] + sorted(df_full[DatasetKeys.BARRIO].unique().tolist())
    barrio_filtro = st.selectbox("Barrio / Contrato", barrios_lista, key="sidebar_barrio")

    st.markdown("#### Feature del Mapa de Calor")
    feature_label = st.radio(
        "Variable a visualizar:",
        list(FEATURES_DISPONIBLES.keys()),
        index=2, # Ratio Consumo/Contrato por defecto
        key="feature_radio",
    )
    feature_col = FEATURES_DISPONIBLES[feature_label]

    st.markdown("#### Filtro por Uso")
    usos_disponibles = ["Todos los usos"] + sorted(df_full[DatasetKeys.USO].unique().tolist())
    # Recomendamos filtro DOMESTICO
    uso_filtro = st.selectbox("Tipo de uso", usos_disponibles, 
                              index=usos_disponibles.index("DOMESTICO") if "DOMESTICO" in usos_disponibles else 0)


    st.markdown("---")
    # Selector de barrio por clic en mapa
    if st.session_state.barrio_seleccionado:
        st.markdown(
            f"""<div style="background:rgba(76,201,240,0.1); border:1px solid #4cc9f0;
            border-radius:8px; padding:10px; text-align:center; font-size:13px;">
            <b>Barrio activo:</b><br>
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
#  FILTRADO DE DATOS
# ════════════════════════════════════════════════════════════════════════════
df_filtered = filter_dataframe(df_full, fecha_inicio, fecha_fin, barrio_filtro, uso_filtro)
df_barrio_agg = aggregate_by_barrio(df_filtered)


# ════════════════════════════════════════════════════════════════════════════
# ENCABEZADO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 10px 0 20px;">
    <h1 style="margin:0; font-size:28px; color:#4cc9f0;">
        INVICTUS — Dashboard de Anomalías Hídricas y Atribución Causal
    </h1>
    <p style="color:#888; font-size:13px; margin-top:4px;">
        Detección de anomalías y presión turística en Alicante · Análisis Físico y Estadístico
    </p>
</div>
""", unsafe_allow_html=True)

# KPIs globales rápidos
total_contratos = int(df_filtered[DatasetKeys.NUM_CONTRATOS].sum()) if DatasetKeys.NUM_CONTRATOS in df_filtered.columns else 0
alert_col_global = "num_alertas"
total_alertas   = int((df_filtered[DatasetKeys.ALERTA_NIVEL] != 'Normal').sum()) if DatasetKeys.ALERTA_NIVEL in df_filtered.columns else 0
total_consumo   = df_filtered[DatasetKeys.CONSUMO].sum() if DatasetKeys.CONSUMO in df_filtered.columns else 0
num_barrios     = df_filtered[DatasetKeys.BARRIO].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Barrios analizados",    num_barrios)
k2.metric("Contratos procesados",  f"{total_contratos:,}")
k3.metric("Consumo total (m³)",    f"{total_consumo:,.0f}")
k4.metric("Alertas activas",       total_alertas,
          delta="Prioridad Alta" if total_alertas > 10 else None,
          delta_color="inverse" if total_alertas > 10 else "off")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════════════════
tab_mapa, tab_whatif, tab_informe, tab_auditoria = st.tabs([
    "Mapa de Calor Interactivo",
    "Simulador What-if",
    "Informe LLM",
    "Auditoría de Bases",
])


# ─── TAB 1: MAPA ────────────────────────────────────────────────────────────
with tab_mapa:
    # UX Mejorada: Layout Horizontal "Above the Fold" para pantallas de ordenador.
    col_mapa, col_panel = st.columns([1.1, 1], gap="large")

    with col_mapa:
        st.markdown(
            f"""<div style="margin-bottom: 5px;">
                  <span style="font-size:18px; font-weight:600; color:#4cc9f0;">Visión Satelital</span>
                  <span style="color:#888; font-size:13px; margin-left:10px;">({feature_label})</span>
                </div>""", 
            unsafe_allow_html=True
        )
        map_output = render_map(df_barrio_agg, feature_col, gdf, alert_col_global)

        # Detectar clic en el mapa
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
            st.markdown(
                f"""<div style="margin-bottom: 5px;">
                      <span style="font-size:18px; font-weight:600; color:#f4a261;">Análisis Físico:</span>
                      <span style="color:#e0e0e0; font-size:18px; font-weight:700; margin-left:5px;">{st.session_state.barrio_seleccionado}</span>
                    </div>""", 
                unsafe_allow_html=True
            )
            
            # Filtramos los datos temporales
            barrios_limpios = df_filtered[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
            df_panel = df_filtered[barrios_limpios == st.session_state.barrio_seleccionado].copy()
            
            if not df_panel.empty and DatasetKeys.FECHA in df_panel.columns:
                if DatasetKeys.ALERTA_NIVEL in df_panel.columns:
                    df_panel['es_alerta'] = (df_panel[DatasetKeys.ALERTA_NIVEL] != "Normal").astype(int)
                
                agg_cols = {}
                for c in [DatasetKeys.CONSUMO, DatasetKeys.CONSUMO_FISICO_ESPERADO]:
                    if c in df_panel.columns: agg_cols[c] = 'sum'
                for c in [DatasetKeys.CONSUMO_RATIO, DatasetKeys.PREDICCION_FOURIER]:
                    if c in df_panel.columns: agg_cols[c] = 'mean'
                
                impact_cols = [
                    DatasetKeys.PCT_CALOR_FRIO, DatasetKeys.PCT_LLUVIA_SEQUIA,
                    DatasetKeys.PCT_VEGETACION, DatasetKeys.PCT_TURISMO,
                    DatasetKeys.PCT_FIESTA, DatasetKeys.PCT_CAUSA_DESCONOCIDA
                ]
                for c in impact_cols:
                    if c in df_panel.columns: agg_cols[c] = 'mean'
                    
                if 'es_alerta' in df_panel.columns:
                    agg_cols['es_alerta'] = 'max'
                # Nivel de alerta y Z-score para el tooltip por punto
                if DatasetKeys.ALERTA_NIVEL in df_panel.columns:
                    agg_cols[DatasetKeys.ALERTA_NIVEL] = lambda x: x.mode()[0] if len(x) > 0 else "Normal"
                if DatasetKeys.Z_ERROR_FINAL in df_panel.columns:
                    agg_cols[DatasetKeys.Z_ERROR_FINAL] = 'mean'
                
                if agg_cols:
                    df_panel_temporal = df_panel.groupby(DatasetKeys.FECHA).agg(agg_cols).reset_index()
                else:
                    df_panel_temporal = df_panel.copy()

                df_panel_temporal = df_panel_temporal.sort_values(DatasetKeys.FECHA)
                
                val_real = df_panel_temporal[DatasetKeys.CONSUMO_RATIO] if DatasetKeys.CONSUMO_RATIO in df_panel_temporal.columns else df_panel_temporal.get(DatasetKeys.CONSUMO, pd.Series([0]*len(df_panel_temporal)))
                val_est  = df_panel_temporal[DatasetKeys.CONSUMO_FISICO_ESPERADO] if DatasetKeys.CONSUMO_FISICO_ESPERADO in df_panel_temporal.columns else df_panel_temporal.get(DatasetKeys.PREDICCION_FOURIER, pd.Series([0]*len(df_panel_temporal)))
                
                import plotly.graph_objects as go
                fig_panel = go.Figure()

                fechas_str = df_panel_temporal[DatasetKeys.FECHA].dt.strftime("%Y-%m")

                # ── Banda verde de normalidad ±1.5σ (Z-Score Sincronizado) ───────
                error_calc = val_real - val_est
                error_std = error_calc.std()
                if pd.isna(error_std) or error_std == 0:
                    error_std = 1.0
                error_mean = error_calc.mean()

                banda_sup = (val_est + error_mean) + (1.5 * error_std)
                banda_inf = ((val_est + error_mean) - (1.5 * error_std)).clip(lower=0)
                fig_panel.add_trace(go.Scatter(
                    x=pd.concat([fechas_str, fechas_str.iloc[::-1]]),
                    y=pd.concat([banda_sup, banda_inf.iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(82,183,136,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Normalidad (Z < 1.5)",
                ))

                # ── Predicción IA (Fourier + RF) ─────────────────────────────────
                fig_panel.add_trace(go.Scatter(
                    x=fechas_str, y=val_est,
                    mode="lines", name="Predicción IA",
                    line=dict(color="#f4a261", width=2, dash="dash"),
                    hovertemplate="<b>%{x}</b><br>Predicción: %{y:.2f} m³/cto<extra></extra>"
                ))

                # ── Consumo Real ──────────────────────────────────────────────────
                fig_panel.add_trace(go.Scatter(
                    x=fechas_str, y=val_real,
                    mode="lines+markers", name="Consumo Real",
                    line=dict(color="#e0e0e0", width=2),
                    marker=dict(size=4, color="#e0e0e0"),
                    hovertemplate="<b>%{x}</b><br>Real: %{y:.2f} m³/cto<extra></extra>"
                ))

                # ── Marcadores de anomalía color-coded por nivel ──────────────────
                if DatasetKeys.ALERTA_NIVEL in df_panel_temporal.columns:
                    col_nivel = df_panel_temporal[DatasetKeys.ALERTA_NIVEL]

                    def _build_hover(row):
                        """Tooltip con nivel + Z-score + porcentajes SHAP de ese mes."""
                        nivel_txt  = str(row.get(DatasetKeys.ALERTA_NIVEL, "—"))
                        z_val      = row.get(DatasetKeys.Z_ERROR_FINAL, float("nan"))
                        p_calor    = row.get(DatasetKeys.PCT_CALOR_FRIO,        0) or 0
                        p_lluvia   = row.get(DatasetKeys.PCT_LLUVIA_SEQUIA,     0) or 0
                        p_ndvi     = row.get(DatasetKeys.PCT_VEGETACION,        0) or 0
                        p_turismo  = row.get(DatasetKeys.PCT_TURISMO,           0) or 0
                        p_fiesta   = row.get(DatasetKeys.PCT_FIESTA,            0) or 0
                        p_desc     = row.get(DatasetKeys.PCT_CAUSA_DESCONOCIDA, 0) or 0
                        z_str = f"{z_val:.2f} σ" if pd.notna(z_val) else "—"
                        return (
                            f"<b>Nivel: {nivel_txt}</b>   Z = {z_str}<br>"
                            f"─────────────────────────<br>"
                            f"Clima Temp.:          {p_calor:.1f}%<br>"
                            f"Clima Preci.:         {p_lluvia:.1f}%<br>"
                            f"Vegetación:           {p_ndvi:.1f}%<br>"
                            f"Turismo (Pernoc.):    {p_turismo:.1f}%<br>"
                            f"Festividades:         {p_fiesta:.1f}%<br>"
                            f"Causa Descon.:        {p_desc:.1f}%"
                        )

                    # 🔴 Exceso Grave (siempre en leyenda)
                    mask_grave = col_nivel.isin(["1_EXCESO_Grave"])
                    if mask_grave.any():
                        hover_g = df_panel_temporal[mask_grave].apply(_build_hover, axis=1).tolist()
                        fig_panel.add_trace(go.Scatter(
                            x=fechas_str[mask_grave], y=val_real[mask_grave],
                            mode="markers", name="Grave",
                            marker=dict(size=14, color="#e74c3c", symbol="circle",
                                        line=dict(width=2, color="#fff")),
                            hovertemplate="%{customdata}<extra></extra>",
                            customdata=hover_g,
                        ))
                    else:
                        fig_panel.add_trace(go.Scatter(
                            x=[], y=[],
                            mode="markers", name="Grave",
                            marker=dict(size=14, color="#e74c3c", symbol="circle", line=dict(width=2, color="#fff")),
                            showlegend=True,
                        ))

                    # 🟠 Exceso Leve / Moderado
                    mask_leve = col_nivel.isin(["2_EXCESO_Moderado", "3_EXCESO_Leve"])
                    if mask_leve.any():
                        hover_l = df_panel_temporal[mask_leve].apply(_build_hover, axis=1).tolist()
                        fig_panel.add_trace(go.Scatter(
                            x=fechas_str[mask_leve], y=val_real[mask_leve],
                            mode="markers", name="Leve/Mod",
                            marker=dict(size=11, color="#f39c12", symbol="circle",
                                        line=dict(width=2, color="#fff")),
                            hovertemplate="%{customdata}<extra></extra>",
                            customdata=hover_l,
                        ))
                    else:
                        fig_panel.add_trace(go.Scatter(
                            x=[], y=[],
                            mode="markers", name="Leve/Mod",
                            marker=dict(size=11, color="#f39c12", symbol="circle", line=dict(width=2, color="#fff")),
                            showlegend=True,
                        ))

                    # 🔵 Defecto (consumo bajo)
                    mask_def = col_nivel.isin(["4_DEFECTO_Grave", "5_DEFECTO_Moderado", "6_DEFECTO_Leve"])
                    if mask_def.any():
                        hover_d = df_panel_temporal[mask_def].apply(_build_hover, axis=1).tolist()
                        fig_panel.add_trace(go.Scatter(
                            x=fechas_str[mask_def], y=val_real[mask_def],
                            mode="markers", name="Defecto",
                            marker=dict(size=11, color="#3498db", symbol="circle",
                                        line=dict(width=2, color="#fff")),
                            hovertemplate="%{customdata}<extra></extra>",
                            customdata=hover_d,
                        ))
                    else:
                        fig_panel.add_trace(go.Scatter(
                            x=[], y=[],
                            mode="markers", name="Defecto",
                            marker=dict(size=11, color="#3498db", symbol="circle", line=dict(width=2, color="#fff")),
                            showlegend=True,
                        ))

                fig_panel.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
                    yaxis=dict(
                        title="Consumo (m³/cto)",
                        gridcolor="rgba(255,255,255,0.05)",
                        rangemode="tozero"
                    ),
                    legend=dict(
                        orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=11)
                    ),
                    height=340,
                    hovermode="closest",
                )

                st.plotly_chart(fig_panel, width='stretch')
                
                # ── Tabla de Alertas del Barrio (CSVs de Riesgos) ────────────────
                import glob

                riesgos_dir = Paths.PROC_CSV_RIESGOS_DIR
                nombres_alertas = {
                    "1_EXCESO_Grave":    ("●", "#e74c3c"),
                    "2_EXCESO_Moderado": ("●", "#e67e22"),
                    "3_EXCESO_Leve":     ("●", "#f39c12"),
                    "4_DEFECTO_Grave":   ("●", "#3498db"),
                    "5_DEFECTO_Moderado":("●", "#5dade2"),
                    "6_DEFECTO_Leve":    ("●", "#85c1e9"),
                }

                barrio_norm = st.session_state.barrio_seleccionado  # ya en upper

                alertas_encontradas = False
                for nivel_key, (emoji, color) in nombres_alertas.items():
                    csv_path = riesgos_dir / f"{nivel_key}.csv"
                    if not csv_path.exists():
                        continue
                    try:
                        df_csv = pd.read_csv(csv_path)
                    except Exception:
                        continue

                    # Filtrar por barrio (normalizar para comparar)
                    if DatasetKeys.BARRIO in df_csv.columns:
                        barrio_col = df_csv[DatasetKeys.BARRIO].astype(str).str.split("-", n=1).str[-1].str.strip().str.upper()
                        df_csv_barrio = df_csv[barrio_col == barrio_norm].copy()
                    else:
                        # Fallback: buscar en todas las columnas de texto
                        df_csv_barrio = df_csv[df_csv.apply(
                            lambda row: any(barrio_norm in str(v).upper() for v in row), axis=1
                        )].copy()

                    if df_csv_barrio.empty:
                        continue

                    # Filtrar por Uso si aplica
                    if DatasetKeys.USO in df_csv_barrio.columns and uso_filtro != "Todos los usos":
                        df_csv_barrio = df_csv_barrio[df_csv_barrio[DatasetKeys.USO] == uso_filtro]

                    # Filtrar por Rango Temporal
                    if DatasetKeys.FECHA in df_csv_barrio.columns and not df_csv_barrio.empty:
                        fechas_csv = pd.to_datetime(df_csv_barrio[DatasetKeys.FECHA], errors='coerce')
                        mask = (fechas_csv >= fecha_inicio) & (fechas_csv <= fecha_fin)
                        df_csv_barrio = df_csv_barrio[mask]
                        
                    if df_csv_barrio.empty:
                        continue

                    alertas_encontradas = True
                    st.markdown(
                        f"""<div style='border-left:3px solid {color}; padding:6px 12px; 
                        margin-bottom:6px; background:rgba(255,255,255,0.03); border-radius:4px;'>
                        <b style='color:{color};'>{emoji} {nivel_key.replace('_',' ')}</b>
                        <span style='color:#888; font-size:12px; margin-left:8px;'>
                        — {len(df_csv_barrio)} registro(s)</span></div>""",
                        unsafe_allow_html=True
                    )
                    # Mostrar columnas relevantes
                    cols_show = [c for c in df_csv_barrio.columns
                                 if not c.startswith('litros_') and not c.startswith('shap_')]
                    st.dataframe(
                        df_csv_barrio[cols_show].reset_index(drop=True),
                        width='stretch',
                        hide_index=True,
                    )

                if not alertas_encontradas:
                    st.success("No se han registrado anomalías en este barrio para el filtro seleccionado.")

            else:
                st.info("No hay datos temporales para visualizar.")
        else:
            # Empty State Elegante
            st.markdown("""
            <div style="
                height: 600px; display: flex; flex-direction: column; align-items: center; justify-content: center; 
                background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%);
                border: 1px dashed rgba(76,201,240,0.3); border-radius: 12px; color: #668; text-align: center;
                padding: 40px;
            ">
                <div style="font-size: 50px; margin-bottom:15px; opacity:0.6;">◈</div>
                <div style="font-size: 18px; color: #a8dadc; font-weight:600; margin-bottom:10px;">
                    Selecciona un Sector
                </div>
                <div style="font-size: 14px; color: #aaa; line-height: 1.5;">
                    Haz clic en cualquier área del mapa interactivo situado a la izquierda para visualizar el estado de consumo físico de un barrio, la trazabilidad de sus anomalías y el diagnóstico de los motores de atribución causal.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 2: WHAT-IF ─────────────────────────────────────────────────────────
with tab_whatif:
    render_whatif(df_filtered, st.session_state.barrio_seleccionado)


# ─── TAB 3: INFORME LLM ────────────────────────────────────────────────────
with tab_informe:
    render_llm_report(st.session_state.barrio_seleccionado, df_filtered)


# ─── TAB 4: AUDITORÍA DE BASES ──────────────────────────────────────────────
with tab_auditoria:
    st.markdown("### Certificación de Cumplimiento")
    st.markdown(
        "<small style='color:#888;'>Este panel detalla cómo el proyecto INVICTUS "
        "se alinea con las Bases de Participación del HACKATHON AMAEM.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.info("#### 1. Originalidad e Innovación (20%)")
        st.markdown("""
        - **Uso no convencional:** Cruce de micro-telelectura con **Gap de Presión Turística** (INE vs Registro GVA).
        - **Modelo Híbrido:** Combinación de series de **Fourier de 2º orden** (física pura) con Random Forest.
        - **Detección Causal:** Atribución automática del % de culpabilidad al clima vs. turismo.
        """)

        st.success("#### 3. Impacto Social/Ambiental (30%)")
        st.markdown("""
        - **Crisis de Vivienda:** Herramienta directa para el análisis de presión turística y sostenibilidad.
        - **Sostenibilidad:** Identificación de sobreconsumos masivos no residenciales en red doméstica.
        - **Transparencia:** Democratización del dato de anomalías hacia los gestores municipales.
        """)

    with c2:
        st.warning("#### 2. Implementación Técnica (30%)")
        st.markdown("""
        - **6 Fuentes de Datos:** AMAEM, INE, GVA, AEMET, Sentinel-2 (Satelital) y Festivos.
        - **Privacidad (Cláusula 13.1):** Procesamiento **100% Local** mediante Ollama + Qwen 7B.
        - **Escalabilidad:** Pipeline modular basado en checkpoints CSV para despliegue en otras ciudades.
        """)

        st.error("#### 4. Presentación y Comunicación (20%)")
        st.markdown("""
        - **Simulador What-If:** Motor de inferencia interactivo con Validación de Mahalanobis.
        - **IA Narrativa:** Transformación de métricas técnicas en informes legibles para decisión.
        - **Visualización Premium:** Mapa choropleth sincronizado y KPIs de impacto en tiempo real.
        """)

    st.markdown("---")
    st.markdown(
        """<div style="background:rgba(255,255,255,0.05); border:1px dashed rgba(76,201,240,0.4);
        border-radius:8px; padding:15px; color:#aaa; font-size:13px;">
        <b>DECLARACIÓN DE CUMPLIMIENTO:</b> Este proyecto ha sido diseñado respetando los principios de 
        <b>Innovación Abierta</b> y el uso de herramientas de acceso libre (Cláusula 6.1). Especialmente, se garantiza 
        el respeto a la confidencialidad de los datos mediante el uso de redes neuronales locales (Ollama), asegurando 
        que <b>ningún dato de telelectura</b> sea transmitido fuera del entorno controlado de ejecución.
        </div>""",
        unsafe_allow_html=True,
    )
