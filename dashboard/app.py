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

# ── Configuración de pandas y ruta para src/ ──────────────────────────────
pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DatasetKeys
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
# CARGA DE DATOS (cacheada)
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("Cargando datos del pipeline Water2Fraud..."):
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
        <div style="font-size:11px; color:#888; margin-top:2px;">Water2Fraud · Detector Turístico</div>
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
        🌊 INVICTUS — Dashboard de Detección de Fraude Turístico
    </h1>
    <p style="color:#888; font-size:13px; margin-top:4px;">
        Detección de viviendas turísticas ilegales en Alicante · Análisis Físico y Estadístico
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
          delta="⚠️" if total_alertas > 10 else None,
          delta_color="inverse" if total_alertas > 10 else "off")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════════════════
tab_mapa, tab_whatif, tab_informe = st.tabs([
    "Mapa de Calor Interactivo",
    "Simulador What-if",
    "Informe LLM",
])


# ─── TAB 1: MAPA ────────────────────────────────────────────────────────────
with tab_mapa:
    # UX Mejorada: Layout Horizontal "Above the Fold" para pantallas de ordenador.
    col_mapa, col_panel = st.columns([1.1, 1], gap="large")

    with col_mapa:
        st.markdown(
            f"""<div style="margin-bottom: 5px;">
                  <span style="font-size:18px; font-weight:600; color:#4cc9f0;">🌍 Visión Satelital</span>
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
                      <span style="font-size:18px; font-weight:600; color:#f4a261;">🔬 Análisis Físico:</span>
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
                
                fig_panel.add_trace(go.Scatter(
                    x=fechas_str, y=val_real,
                    mode="lines+markers", name="Real (m³/cto)",
                    line=dict(color="#4cc9f0", width=2)
                ))
                
                fig_panel.add_trace(go.Scatter(
                    x=fechas_str, y=val_est,
                    mode="lines", name="Estimado",
                    line=dict(color="#f4a261", width=2, dash="dash")
                ))

                if 'es_alerta' in df_panel_temporal.columns:
                    mask_alertas = df_panel_temporal['es_alerta'] > 0
                    if mask_alertas.any():
                        fig_panel.add_trace(go.Scatter(
                            x=fechas_str[mask_alertas], y=val_real[mask_alertas],
                            mode="markers", name="Anomalía Detectada",
                            marker=dict(size=14, symbol="circle-open", line=dict(width=3, color="#e74c3c")),
                            hoverinfo="skip"
                        ))
                
                fig_panel.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", rangemode="tozero"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                    height=300, hovermode="x unified"
                )
                
                st.plotly_chart(fig_panel, width="stretch")
                
                # Desglose de anomalías con KPIs rápidos
                if 'es_alerta' in df_panel_temporal.columns and df_panel_temporal['es_alerta'].any():
                    alertas_activas = int((df_panel[DatasetKeys.ALERTA_NIVEL] != "Normal").sum())
                    st.markdown(
                        f"""<div style='background:rgba(231,76,60,0.1); border-left:3px solid #e74c3c; padding:10px 15px; border-radius:4px; margin-bottom:15px;'>
                            <span style='color:#e74c3c; font-weight:700;'>{alertas_activas} ALERTAS ACTIVAS</span> 
                            <span style='color:#aaa; font-size:13px;'> — Promedio de causalidad detectada:</span>
                        </div>""", unsafe_allow_html=True
                    )
                    
                    df_alertas = df_panel_temporal[df_panel_temporal['es_alerta'] > 0]
                    impact_data_raw = {
                        "Clima Temp.":      df_alertas.get(DatasetKeys.PCT_CALOR_FRIO,      pd.Series([0])).mean(),
                        "Clima Preci.":     df_alertas.get(DatasetKeys.PCT_LLUVIA_SEQUIA,   pd.Series([0])).mean(),
                        "Vegetación":       df_alertas.get(DatasetKeys.PCT_VEGETACION,      pd.Series([0])).mean(),
                        "Turismo Ilegal":   df_alertas.get(DatasetKeys.PCT_TURISMO,         pd.Series([0])).mean(),
                        "Festividades":     df_alertas.get(DatasetKeys.PCT_FIESTA,          pd.Series([0])).mean(),
                    }
                    pct_desconocida = df_alertas.get(DatasetKeys.PCT_CAUSA_DESCONOCIDA, pd.Series([0])).mean()
                    pct_desconocida = pct_desconocida if not pd.isna(pct_desconocida) else 0.0
                    pct_conocida = 100.0 - pct_desconocida

                    # -- Indicador de Señal de Fraude (barra prominente) ----------
                    if pct_desconocida >= 80:
                        fraud_color = "#e74c3c"
                        fraud_emoji = "🔴"
                        fraud_label = "SEÑAL FUERTE DE FRAUDE"
                    elif pct_desconocida >= 50:
                        fraud_color = "#e67e22"
                        fraud_emoji = "🟠"
                        fraud_label = "SEÑAL MODERADA"
                    else:
                        fraud_color = "#52b788"
                        fraud_emoji = "🟢"
                        fraud_label = "FACTORES CONOCIDOS PREDOMINAN"

                    st.markdown(
                        f"""
                        <div style="background:rgba({
                            '231,76,60' if pct_desconocida>=80 else
                            '230,126,34' if pct_desconocida>=50 else
                            '82,183,136'
                        },0.12); border:1px solid {fraud_color}; border-radius:10px; padding:14px 18px; margin-bottom:14px;">
                            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                                <span style="font-size:20px;">{fraud_emoji}</span>
                                <span style="color:{fraud_color}; font-weight:700; font-size:15px;">{fraud_label}</span>
                            </div>
                            <div style="font-size:12px; color:#bbb; margin-bottom:10px;">
                                <b style="color:#fff; font-size:18px;">{pct_desconocida:.1f}%</b> del consumo anómalo 
                                no puede atribuirse a factores externos conocidos (clima, turismo, festivos).<br>
                                <span style="color:#aaa;">Un porcentaje alto de inexplicabilidad es la principal señal de VT ilegal.</span>
                            </div>
                            <!-- Barra de progreso bicolor -->
                            <div style="background:rgba(255,255,255,0.08); border-radius:6px; height:12px; overflow:hidden;">
                                <div style="width:{pct_conocida:.1f}%; background:linear-gradient(90deg,#4cc9f0,#52b788); height:100%; float:left; border-radius:6px 0 0 6px;"></div>
                                <div style="width:{pct_desconocida:.1f}%; background:{fraud_color}; height:100%; float:left; opacity:0.7; border-radius:0 6px 6px 0;"></div>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:10px; color:#888; margin-top:4px;">
                                <span>✅ {pct_conocida:.1f}% Explicado por factores conocidos</span>
                                <span>{fraud_emoji} {pct_desconocida:.1f}% Inexplicado (señal fraude)</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # -- Gráfico horizontal de factores conocidos -----------------
                    known_items = {k: v for k, v in impact_data_raw.items()
                                   if not pd.isna(v) and v > 0.05}
                    if known_items:
                        fig_bar_h = go.Figure(go.Bar(
                            x=list(known_items.values()),
                            y=list(known_items.keys()),
                            orientation='h',
                            marker_color=["#f39c12","#4cc9f0","#52b788","#e74c3c","#9b59b6"][:len(known_items)],
                            text=[f"{v:.1f}%" for v in known_items.values()],
                            textposition="outside",
                            textfont=dict(color="#e0e0e0", size=11),
                        ))
                        fig_bar_h.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.02)",
                            margin=dict(l=0, r=60, t=8, b=0),
                            xaxis=dict(title="% atribuido", range=[0, max(known_items.values())*1.4],
                                       gridcolor="rgba(255,255,255,0.06)"),
                            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                            height=180,
                            showlegend=False,
                            title=dict(text="Factores conocidos identificados", font=dict(size=12, color="#aaa"),
                                       x=0, pad=dict(b=4))
                        )
                        st.plotly_chart(fig_bar_h, use_container_width=True)
                else:
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
                <div style="font-size: 50px; margin-bottom:15px; opacity:0.6;">🚰</div>
                <div style="font-size: 18px; color: #a8dadc; font-weight:600; margin-bottom:10px;">
                    Selecciona un Sector
                </div>
                <div style="font-size: 14px; color: #aaa; line-height: 1.5;">
                    Haz clic en cualquier área del mapa interactivo situado a la izquierda para visualizar el estado de consumo físico de un barrio, la trazabilidad de sus anomalías y el diagnóstico de los motores de fraude turístico.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 2: WHAT-IF ─────────────────────────────────────────────────────────
with tab_whatif:
    render_whatif(df_filtered, st.session_state.barrio_seleccionado)


# ─── TAB 3: INFORME LLM ────────────────────────────────────────────────────
with tab_informe:
    render_llm_report(st.session_state.barrio_seleccionado, df_filtered)
