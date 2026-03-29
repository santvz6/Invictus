"""
whatif_simulator.py
-------------------
Simulador interactivo de tipo "What-if Analysis".
El usuario mueve sliders de features y ve en tiempo real cómo varía
el consumo estimado y el riesgo de anomalía del barrio seleccionado.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.config import DatasetKeys, Paths

@st.cache_resource(show_spinner="Cargando motores IA y Físico (Caché)...")
def get_simulation_context():
    """
    Carga los motores físicos entrenados para la simulación interactiva.
    """
    if not Paths.PROC_MODEL_RF.exists() or not Paths.PROC_FEATURES_RF.exists():
        raise FileNotFoundError("No se encontró el modelo de producción. Ejecute 'python main.py --run' primero.")
        
    # 1. Cargar Modelos Físicos (Producción)
    rf_model = joblib.load(Paths.PROC_MODEL_RF)
    
    with open(Paths.PROC_FEATURES_RF, "r") as f:
        features_rf = json.load(f)
        
    context_cols = [c for c in features_rf if c.startswith(DatasetKeys.USO + '_')]
    
    return rf_model, features_rf, context_cols


def render_whatif(df: pd.DataFrame, barrio: str | None = None):
    """
    Renderiza el panel simulador.

    Parameters
    ----------
    df : pd.DataFrame  — DataFrame completo para calcular rangos reales
    barrio : str | None — Si se pasa, muestra los valores reales como referencia
    """
    st.markdown("### Simulador What-if Avanzado (IA + Física)")
    st.markdown(
        "<small style='color:#888;'>Ajusta las variables para ver cómo cambia el consumo esperado "
        "y la respuesta de los modelos predictivos.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    if not barrio:
        barrios_disponibles = df[DatasetKeys.BARRIO].unique()
        barrio = "PLAYA SAN JUAN" if "PLAYA SAN JUAN" in barrios_disponibles else barrios_disponibles[0]
        st.info(f"Selecciona un barrio en el mapa. Mostrando simulación para: **{barrio}**")

    # ─── Filtrar datos del barrio ───────────────────────────────────────
    barrios_limpios = df[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
    df_b = df[(barrios_limpios == barrio)].copy()
    
    uso_col = DatasetKeys.USO
    if uso_col in df_b.columns and "DOMESTICO" in df_b[uso_col].values:
        df_b = df_b[df_b[uso_col] == "DOMESTICO"].copy()
        
    df_b = df_b.sort_values(DatasetKeys.FECHA).tail(12)

    if len(df_b) < 12:
        st.warning(f"No hay suficientes datos (12 meses) para simular el barrio {barrio}.")
        return
        
    df_b[DatasetKeys.FECHA] = pd.to_datetime(df_b[DatasetKeys.FECHA])

    # ─── Carga de Contexto IA ───────────────────────────────────────────
    try:
        rf_model, features_rf, context_cols = get_simulation_context()
    except Exception as e:
        st.error(f"Error cargando contexto de simulación: {e}")
        return

    # ─── Rangos reales del dataset ───────────────────────────────────────
    def _safe_range(col, default_lo, default_hi):
        if col in df.columns:
            lo, hi = float(df[col].min()), float(df[col].max())
            return (lo, hi) if lo < hi else (default_lo, default_hi)
        return (default_lo, default_hi)

    temp_range     = _safe_range(DatasetKeys.TEMP_MEDIA,           5.0, 40.0)
    precip_range   = _safe_range(DatasetKeys.PRECIPITACION,        0.0, 150.0)
    ndvi_range     = _safe_range(DatasetKeys.NDVI_SATELITE,        0.0, 1.0)
    vt_sin_range   = _safe_range(DatasetKeys.PCT_VT_SIN_REGISTRAR, 0.0, 40.0)
    festivos_range = _safe_range(DatasetKeys.PCT_FESTIVOS,         0.0, 15.0)
    ratio_range    = _safe_range(DatasetKeys.CONSUMO_RATIO,        0.1, 10.0)
    
    def_temp     = df_b[DatasetKeys.TEMP_MEDIA].mean()
    def_precip   = df_b[DatasetKeys.PRECIPITACION].mean()
    def_ndvi     = df_b[DatasetKeys.NDVI_SATELITE].mean()
    def_vt_sin   = df_b[DatasetKeys.PCT_VT_SIN_REGISTRAR].mean() if DatasetKeys.PCT_VT_SIN_REGISTRAR in df_b.columns else 0.0
    def_festivos = df_b[DatasetKeys.PCT_FESTIVOS].mean() if DatasetKeys.PCT_FESTIVOS in df_b.columns else 0.0
    def_ratio    = df_b[DatasetKeys.CONSUMO_RATIO].mean()
    

    # ─── Sliders ─────────────────────────────────────────────────────────
    st.markdown("#### Modificación de Features")
    st.caption("Los cambios aplicados se suman o restan uniformemente a la serie histórica de los últimos 12 meses.")
    col1, col2, col3 = st.columns(3)

    with col1:
        temp_val = st.slider(
            "Temperatura Media (°C)",
            min_value=float(temp_range[0]), max_value=float(temp_range[1]),
            value=float(def_temp), step=0.5, key="wif_temp",
        )
        precip_val = st.slider(
            "Precipitación (mm)",
            min_value=float(precip_range[0]), max_value=float(precip_range[1]),
            value=float(def_precip), step=1.0, key="wif_precip",
        )
        ndvi_val = st.slider(
            "Índice Vegetación (NDVI)",
            min_value=float(ndvi_range[0]), max_value=float(ndvi_range[1]),
            value=float(def_ndvi), step=0.05, key="wif_ndvi",
        )

    with col2:
        vt_sin_val = st.slider(
            "% VT Sin Registrar (Ilegales)",
            min_value=float(vt_sin_range[0]), max_value=float(vt_sin_range[1]),
            value=float(def_vt_sin), step=0.5, key="wif_vtsin",
        )
        festivos_val = st.slider(
            "Festividades (% mes)",
            min_value=float(festivos_range[0]), max_value=float(festivos_range[1]),
            value=float(def_festivos), step=0.5, key="wif_festivos",
        )
        
    with col3:
        ratio_val = st.slider(
            "Ratio Consumo Modificado",
            min_value=float(ratio_range[0]), max_value=float(ratio_range[1]),
            value=float(def_ratio), step=0.1, key="wif_ratio",
        )

    # ─── Aplicar deltas a la serie temporal ──────────────────────────────
    df_sim = df_b.copy()
    
    # Modificación Multiplicativa/Aditiva (Mayor rigor físico en What-If)
    if DatasetKeys.TEMP_MEDIA in df_sim.columns: 
        if def_temp != 0: df_sim[DatasetKeys.TEMP_MEDIA] *= (temp_val / def_temp)
        else: df_sim[DatasetKeys.TEMP_MEDIA] += (temp_val - def_temp)
    if DatasetKeys.PRECIPITACION in df_sim.columns: 
        if def_precip != 0: df_sim[DatasetKeys.PRECIPITACION] *= (precip_val / def_precip)
        else: df_sim[DatasetKeys.PRECIPITACION] = np.clip(df_sim[DatasetKeys.PRECIPITACION] + (precip_val - def_precip), 0, None)
    if DatasetKeys.PCT_VT_SIN_REGISTRAR in df_sim.columns: 
        df_sim[DatasetKeys.PCT_VT_SIN_REGISTRAR] = np.clip(df_sim[DatasetKeys.PCT_VT_SIN_REGISTRAR] + (vt_sin_val - def_vt_sin), 0, 100)
    if DatasetKeys.NDVI_SATELITE in df_sim.columns: 
        if def_ndvi != 0: df_sim[DatasetKeys.NDVI_SATELITE] = np.clip(df_sim[DatasetKeys.NDVI_SATELITE] * (ndvi_val / def_ndvi), 0, 1)
        else: df_sim[DatasetKeys.NDVI_SATELITE] = np.clip(df_sim[DatasetKeys.NDVI_SATELITE] + (ndvi_val - def_ndvi), 0, 1)
    if DatasetKeys.PCT_FESTIVOS in df_sim.columns: 
        if def_festivos != 0: df_sim[DatasetKeys.PCT_FESTIVOS] *= (festivos_val / def_festivos)
        else: df_sim[DatasetKeys.PCT_FESTIVOS] = np.clip(df_sim[DatasetKeys.PCT_FESTIVOS] + (festivos_val - def_festivos), 0, 100)
    if DatasetKeys.CONSUMO_RATIO in df_sim.columns: 
        if def_ratio != 0: df_sim[DatasetKeys.CONSUMO_RATIO] = np.clip(df_sim[DatasetKeys.CONSUMO_RATIO] * (ratio_val / def_ratio), 0.1, None)
        else: df_sim[DatasetKeys.CONSUMO_RATIO] = np.clip(df_sim[DatasetKeys.CONSUMO_RATIO] + (ratio_val - def_ratio), 0.1, None)

    # ─── 1. Inferencia del Modelo Físico (RF) ────────────────────────────
    # Aplicamos el mismo preprocesamiento que en ModeloFisico._calculate_ml_impact
    df_sim_ml = pd.get_dummies(df_sim, columns=[DatasetKeys.USO])
    
    # Asegurar que todas las columnas de contexto de USO existen
    for c in context_cols:
        if c not in df_sim_ml.columns:
            df_sim_ml[c] = 0
            
    # La Predicción Fourier ya viene en el DF base que recibe render_whatif
    # Si no existiera (fallback), usamos el consumo como base
    if DatasetKeys.PREDICCION_FOURIER not in df_sim_ml.columns:
        df_sim_ml[DatasetKeys.PREDICCION_FOURIER] = df_sim_ml[DatasetKeys.CONSUMO_RATIO]
        
    X_sim = df_sim_ml[features_rf].fillna(0)
    impacto_sim = rf_model.predict(X_sim)
    consumo_fisico_sim = df_sim_ml[DatasetKeys.PREDICCION_FOURIER] + impacto_sim

    # ─── Métricas resultado ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Estimación Resultante (Último mes simulado)")

    ratio_sim = df_sim[DatasetKeys.CONSUMO_RATIO].iloc[-1]
    ratio_orig = df_b[DatasetKeys.CONSUMO_RATIO].iloc[-1]
    delta_ratio_pct = ((ratio_sim / ratio_orig) - 1) * 100 if ratio_orig > 0 else 0
    
    fisico_sim = consumo_fisico_sim.iloc[-1]
    fisico_orig = df_b[DatasetKeys.CONSUMO_FISICO_ESPERADO].iloc[-1] if DatasetKeys.CONSUMO_FISICO_ESPERADO in df_b.columns else fisico_sim
    delta_fisico_pct = ((fisico_sim / fisico_orig) - 1) * 100 if fisico_orig > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Ratio Consumo Simulado", f"{ratio_sim:.2f}",
              delta=f"{delta_ratio_pct:+.1f}% vs real")
    m2.metric("Esperado Físico (IA-RF)", f"{fisico_sim:.2f}",
              delta=f"{delta_fisico_pct:+.1f}% vs base",
              delta_color="inverse" if delta_fisico_pct > 10 else "normal")
    m3.metric("Riesgo Anomalía (Físico)", f"{delta_fisico_pct:.0f}%",
              delta="SOCIAL (Alerta)" if delta_fisico_pct > 15 else "NORMAL",
              delta_color="inverse" if delta_fisico_pct > 15 else "off")

    # ─── Curva de simulación ──────────────────────────────────────────────
    st.markdown("#### Evolución a 12 meses: Modelos Predictivos en Escenario")
    
    # Abstracción a un escenario simulado base (Año 2024)
    fechas_str = []
    año_base = 2024
    mes_anterior = -1
    for dt in df_sim[DatasetKeys.FECHA]:
        if mes_anterior != -1 and dt.month < mes_anterior:
            año_base += 1 # Cambio de año (ej. Diciembre -> Enero)
        fechas_str.append(f"{año_base}-{dt.month:02d}")
        mes_anterior = dt.month
    
    fig = go.Figure()
    
    # Área base de Ratio Esperado Físico
    fig.add_trace(go.Scatter(
        x=fechas_str, y=consumo_fisico_sim,
        mode="lines", name="Esperado Físico (IA-RF)",
        line=dict(color="#52b788", width=2, dash="dash"),
        fill='tozeroy', fillcolor='rgba(82, 183, 136, 0.05)',
        line_shape='spline'
    ))
    
    # (Omitimos la reconstrucción del Autoencoder)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, type='category'),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Ratio m³ / contrato"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
        hovermode="x unified"
    )
    st.plotly_chart(fig, width='stretch')

    # ─── Gráfico de Importancia (RF) ───────────────────────────────
    with st.expander("Ver Importancia de Factores Físicos", expanded=False):
        importances = rf_model.feature_importances_
        feat_names_friendly = {
            DatasetKeys.TEMP_MEDIA: "Temperatura Media",
            DatasetKeys.PRECIPITACION: "Precipitación",
            DatasetKeys.PCT_VT_SIN_REGISTRAR: "Pisos Ilegales (% VT)",
            DatasetKeys.NDVI_SATELITE: "Índice Vegetación (NDVI)",
            DatasetKeys.PCT_FESTIVOS: "Festividades (% mes)"
        }
        
        imp_dict = {}
        for i, feat in enumerate(features_rf):
            if feat in feat_names_friendly:
                imp_dict[feat_names_friendly[feat]] = importances[i] * 100
                
        sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        if sorted_imp:
            fig2 = go.Figure(go.Bar(
                x=[v for k, v in sorted_imp],
                y=[k for k, v in sorted_imp],
                orientation="h",
                marker_color="#4cc9f0",
                text=[f"{v:.1f}%" for k, v in sorted_imp],
                textposition="outside",
            ))
            fig2.update_layout(
                title="Peso relativo en la predicción física",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                margin=dict(l=0, r=10, t=30, b=0),
                xaxis=dict(title="%", gridcolor="rgba(255,255,255,0.05)", range=[0, max([v for k,v in sorted_imp])*1.2]),
                yaxis=dict(autorange="reversed"),
                height=220,
            )
            st.plotly_chart(fig2, width='stretch')
