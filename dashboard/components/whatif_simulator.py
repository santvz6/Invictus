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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.config import DatasetKeys, Paths
from src.water2fraud.features.preprocessor import WaterPreprocessor
from src.water2fraud.models.autoencoder import LSTMAutoencoder

@st.cache_resource(show_spinner="Cargando motores IA y Físico...")
def get_simulation_context():
    """
    Carga los datos en bruto y entrena/ajusta los modelos físicos y escaladores 
    necesarios para la simulación interactiva fiel al pipeline original.
    """
    # 1. Carga de datos base no escalados
    df_raw = pd.read_csv(Paths.PROC_CSV_AMAEM_NOT_SCALED)
    df_raw[DatasetKeys.FECHA] = pd.to_datetime(df_raw[DatasetKeys.FECHA], errors='coerce')
    
    # Integrar baseline Fourier si está disponible
    if Paths.PROC_CSV_AMAEM_FISICOS.exists():
        df_fis = pd.read_csv(Paths.PROC_CSV_AMAEM_FISICOS)
        df_fis[DatasetKeys.FECHA] = pd.to_datetime(df_fis[DatasetKeys.FECHA], errors='coerce')
        df_raw = pd.merge(
            df_raw, 
            df_fis[[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA, DatasetKeys.PREDICCION_FOURIER]], 
            on=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA], 
            how='left'
        )
    else:
        df_raw[DatasetKeys.PREDICCION_FOURIER] = df_raw[DatasetKeys.CONSUMO_RATIO] * 0.95
        
    df_raw[DatasetKeys.PREDICCION_FOURIER] = df_raw[DatasetKeys.PREDICCION_FOURIER].fillna(df_raw[DatasetKeys.CONSUMO_RATIO] * 0.95)
    
    # 2. Entrenar el Modelo Físico (Random Forest de impacto exógeno)
    df_ml = pd.get_dummies(df_raw, columns=[DatasetKeys.USO])
    exogenas = [c for c in WaterPreprocessor.FEATURES.keys() if c in df_ml.columns and c != DatasetKeys.CONSUMO_RATIO]
    if DatasetKeys.MES not in df_ml.columns:
        df_ml[DatasetKeys.MES] = df_ml[DatasetKeys.FECHA].dt.month
        
    df_ml['mes_temp'] = df_ml[DatasetKeys.FECHA].dt.month
    context_cols = [c for c in df_ml.columns if c.startswith(DatasetKeys.USO + '_')]
    
    features_rf = exogenas + [DatasetKeys.PREDICCION_FOURIER, 'mes_temp'] + context_cols
    X_rf = df_ml[features_rf].fillna(0)
    y_rf = (df_ml[DatasetKeys.CONSUMO_RATIO] - df_ml[DatasetKeys.PREDICCION_FOURIER]).fillna(0)
    
    rf_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    rf_model.fit(X_rf, y_rf)
    
    # 3. Ajustar Escaladores para el Autoencoder
    scalers = {}
    for col, scale_type in WaterPreprocessor.FEATURES.items():
        if col in df_raw.columns:
            if scale_type == WaterPreprocessor.ROBUST:
                s = RobustScaler()
                s.fit(df_raw[[col]])
                scalers[col] = s
            elif scale_type == WaterPreprocessor.MIN_MAX:
                s = MinMaxScaler()
                s.fit(df_raw[[col]])
                scalers[col] = s
                
    return rf_model, scalers, features_rf, context_cols


def render_whatif(df: pd.DataFrame, barrio: str | None = None):
    """
    Renderiza el panel simulador.

    Parameters
    ----------
    df : pd.DataFrame  — DataFrame completo para calcular rangos reales
    barrio : str | None — Si se pasa, muestra los valores reales como referencia
    """
    st.markdown("### 🔬 Simulador What-if Avanzado (IA + Física)")
    st.markdown(
        "<small style='color:#888;'>Ajusta las variables para ver cómo cambia el consumo esperado "
        "y la respuesta de los modelos predictivos.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    if not barrio:
        barrios_disponibles = df[DatasetKeys.BARRIO].unique()
        barrio = "PLAYA SAN JUAN" if "PLAYA SAN JUAN" in barrios_disponibles else barrios_disponibles[0]
        st.info(f"👆 Selecciona un barrio en el mapa. Mostrando simulación para: **{barrio}**")

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
        rf_model, scalers, features_rf, context_cols = get_simulation_context()
    except Exception as e:
        st.error(f"Error cargando contexto de simulación: {e}")
        return

    # ─── Rangos reales del dataset ───────────────────────────────────────
    def _safe_range(col, default_lo, default_hi):
        if col in df.columns:
            lo, hi = float(df[col].min()), float(df[col].max())
            return (lo, hi) if lo < hi else (default_lo, default_hi)
        return (default_lo, default_hi)

    temp_range    = _safe_range(DatasetKeys.TEMP_MEDIA,      5.0,  40.0)
    precip_range  = _safe_range(DatasetKeys.PRECIPITACION,   0.0, 150.0)
    vt_range      = _safe_range(DatasetKeys.PCT_VT_BARRIO_INE,   0.0, 80.0)
    ratio_range   = _safe_range(DatasetKeys.CONSUMO_RATIO,   0.1,  30.0)
    vt_sin_range  = _safe_range(DatasetKeys.PCT_VT_SIN_REGISTRAR, 0.0, 50.0)
    ndvi_range    = _safe_range(DatasetKeys.NDVI_SATELITE, 0.0, 1.0)
    hoteles_range = _safe_range(DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA, 0.0, 5000.0)

    # Valores por defecto: media del barrio en los últimos 12 meses
    def_temp   = df_b[DatasetKeys.TEMP_MEDIA].mean() if DatasetKeys.TEMP_MEDIA in df_b.columns else np.mean(temp_range)
    def_precip = df_b[DatasetKeys.PRECIPITACION].mean() if DatasetKeys.PRECIPITACION in df_b.columns else np.mean(precip_range)
    def_vt     = df_b[DatasetKeys.PCT_VT_BARRIO_INE].mean() if DatasetKeys.PCT_VT_BARRIO_INE in df_b.columns else np.mean(vt_range)
    def_ratio  = df_b[DatasetKeys.CONSUMO_RATIO].mean() if DatasetKeys.CONSUMO_RATIO in df_b.columns else np.mean(ratio_range)
    def_vt_sin = df_b[DatasetKeys.PCT_VT_SIN_REGISTRAR].mean() if DatasetKeys.PCT_VT_SIN_REGISTRAR in df_b.columns else np.mean(vt_sin_range)
    def_ndvi   = df_b[DatasetKeys.NDVI_SATELITE].mean() if DatasetKeys.NDVI_SATELITE in df_b.columns else np.mean(ndvi_range)
    def_hoteles= df_b[DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA].mean() if DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA in df_b.columns else np.mean(hoteles_range)

    # ─── Sliders ─────────────────────────────────────────────────────────
    st.markdown("#### 🎛️ Modificación de Features")
    st.caption("Los cambios aplicados se suman o restan uniformemente a la serie histórica de los últimos 12 meses.")
    col1, col2, col3 = st.columns(3)

    with col1:
        temp_val = st.slider(
            "🌡 Temperatura Media (°C)",
            min_value=temp_range[0], max_value=temp_range[1],
            value=float(def_temp), step=0.5, key="wif_temp",
        )
        precip_val = st.slider(
            "🌧 Precipitación (mm)",
            min_value=precip_range[0], max_value=precip_range[1],
            value=float(def_precip), step=1.0, key="wif_precip",
        )
        ndvi_val = st.slider(
            "🌿 Índice Vegetación (NDVI)",
            min_value=float(ndvi_range[0]), max_value=float(ndvi_range[1]),
            value=float(def_ndvi), step=0.05, key="wif_ndvi",
        )

    with col2:
        vt_val = st.slider(
            "🏖 % VT Barrio Oficial",
            min_value=float(vt_range[0]), max_value=float(vt_range[1]),
            value=float(def_vt), step=0.5, key="wif_vt",
        )
        vt_sin_val = st.slider(
            "🕵️ % VT Sin Registrar (Ilegales)",
            min_value=float(vt_sin_range[0]), max_value=float(vt_sin_range[1]),
            value=float(def_vt_sin), step=0.5, key="wif_vtsin",
        )
        hoteles_val = st.slider(
            "🏨 Plazas Hoteleras",
            min_value=float(hoteles_range[0]), max_value=float(hoteles_range[1]),
            value=float(def_hoteles), step=10.0, key="wif_hoteles",
        )
        
    with col3:
        ratio_val = st.slider(
            "📊 Ratio Consumo Modificado",
            min_value=ratio_range[0], max_value=ratio_range[1],
            value=float(def_ratio), step=0.1, key="wif_ratio",
        )

    # ─── Aplicar deltas a la serie temporal ──────────────────────────────
    df_sim = df_b.copy()
    
    if DatasetKeys.TEMP_MEDIA in df_sim.columns: 
        df_sim[DatasetKeys.TEMP_MEDIA] += (temp_val - def_temp)
    if DatasetKeys.PRECIPITACION in df_sim.columns: 
        df_sim[DatasetKeys.PRECIPITACION] = np.clip(df_sim[DatasetKeys.PRECIPITACION] + (precip_val - def_precip), 0, None)
    if DatasetKeys.PCT_VT_BARRIO_INE in df_sim.columns: 
        df_sim[DatasetKeys.PCT_VT_BARRIO_INE] = np.clip(df_sim[DatasetKeys.PCT_VT_BARRIO_INE] + (vt_val - def_vt), 0, 100)
    if DatasetKeys.PCT_VT_SIN_REGISTRAR in df_sim.columns: 
        df_sim[DatasetKeys.PCT_VT_SIN_REGISTRAR] = np.clip(df_sim[DatasetKeys.PCT_VT_SIN_REGISTRAR] + (vt_sin_val - def_vt_sin), 0, 100)
    if DatasetKeys.NDVI_SATELITE in df_sim.columns: 
        df_sim[DatasetKeys.NDVI_SATELITE] = np.clip(df_sim[DatasetKeys.NDVI_SATELITE] + (ndvi_val - def_ndvi), 0, 1)
    if DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA in df_sim.columns: 
        df_sim[DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA] = np.clip(df_sim[DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA] + (hoteles_val - def_hoteles), 0, None)
    if DatasetKeys.CONSUMO_RATIO in df_sim.columns: 
        df_sim[DatasetKeys.CONSUMO_RATIO] = np.clip(df_sim[DatasetKeys.CONSUMO_RATIO] + (ratio_val - def_ratio), 0.1, None)

    # ─── 1. Inferencia del Modelo Físico (RF) ────────────────────────────
    df_sim_ml = pd.get_dummies(df_sim, columns=[DatasetKeys.USO])
    for c in context_cols:
        if c not in df_sim_ml.columns:
            df_sim_ml[c] = 0
            
    if DatasetKeys.MES not in df_sim_ml.columns:
        df_sim_ml[DatasetKeys.MES] = df_sim_ml[DatasetKeys.FECHA].dt.month
    df_sim_ml['mes_temp'] = df_sim_ml[DatasetKeys.FECHA].dt.month
    
    if DatasetKeys.PREDICCION_FOURIER not in df_sim_ml.columns:
        df_sim_ml[DatasetKeys.PREDICCION_FOURIER] = df_sim_ml[DatasetKeys.CONSUMO_RATIO] * 0.95
        
    X_sim = df_sim_ml[features_rf].fillna(0)
    impacto_sim = rf_model.predict(X_sim)
    consumo_fisico_sim = df_sim_ml[DatasetKeys.PREDICCION_FOURIER] + impacto_sim

    # ─── 2. Inferencia del Modelo de IA (LSTM-AE) ────────────────────────
    seq_scaled = []
    # Extraemos SIEMPRE todas las features en el orden exacto del entrenamiento
    feature_cols = list(WaterPreprocessor.FEATURES.keys())
    
    for col in feature_cols:
        if col == DatasetKeys.MES_SIN:
            meses = df_sim[DatasetKeys.FECHA].dt.month
            val = ((np.sin(2 * np.pi * meses / 12) + 1) / 2).values.reshape(-1, 1)
            seq_scaled.append(val)
        elif col == DatasetKeys.MES_COS:
            meses = df_sim[DatasetKeys.FECHA].dt.month
            val = ((np.cos(2 * np.pi * meses / 12) + 1) / 2).values.reshape(-1, 1)
            seq_scaled.append(val)
        elif col in scalers and col in df_sim.columns:
            val = scalers[col].transform(df_sim[[col]])
            seq_scaled.append(val)
        else:
            # Relleno de seguridad para mantener la dimensión (no debería ocurrir en core features)
            val = np.zeros((len(df_sim), 1))
            seq_scaled.append(val)
                
    seq_tensor = np.hstack(seq_scaled)
    
    cluster_id = df_b[DatasetKeys.CLUSTER].iloc[0] if DatasetKeys.CLUSTER in df_b.columns else 0
    model_ae = None
    
    if Paths.EXPERIMENTS_DIR.exists():
        exp_dirs = sorted([d for d in Paths.EXPERIMENTS_DIR.iterdir() if d.is_dir()])
        if exp_dirs:
            model_path = exp_dirs[-1] / f"ae_cluster_{int(cluster_id)}.pth"
            if model_path.exists():
                model_ae = LSTMAutoencoder(num_features=len(feature_cols), hidden_dim=128, latent_dim=16, seq_len=12)
                try:
                    model_ae.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                except TypeError:
                    model_ae.load_state_dict(torch.load(model_path, map_location="cpu"))
                model_ae.eval()

    if model_ae:
        with torch.no_grad():
            t_in = torch.tensor(seq_tensor, dtype=torch.float32).unsqueeze(0)
            reconst = model_ae(t_in).squeeze(0).numpy()
            
        # Desescalar el ratio reconstruido
        if DatasetKeys.CONSUMO_RATIO in feature_cols:
            idx_ratio = feature_cols.index(DatasetKeys.CONSUMO_RATIO)
            reconst_ratio_scaled = reconst[:, idx_ratio].reshape(-1, 1)
            consumo_ae_sim = scalers[DatasetKeys.CONSUMO_RATIO].inverse_transform(reconst_ratio_scaled).flatten()
        else:
            consumo_ae_sim = df_sim[DatasetKeys.CONSUMO_RATIO].values
            
        # Calcular puntuación de anomalía estandarizada (AE_SCORE)
        ae_mae = np.mean(np.abs(reconst - seq_tensor))
        error_orig = df_b[DatasetKeys.RECONSTRUCTION_ERROR].iloc[-1] if DatasetKeys.RECONSTRUCTION_ERROR in df_b.columns else 0.1
        score_orig = df_b[DatasetKeys.AE_SCORE].iloc[-1] if DatasetKeys.AE_SCORE in df_b.columns else 50.0
        
        umbral = error_orig / (score_orig / 100) if score_orig > 0 else 0.1
        if umbral == 0: umbral = 0.1
        
        ae_score_sim = (ae_mae / umbral) * 100
    else:
        consumo_ae_sim = df_sim[DatasetKeys.CONSUMO_RATIO].values
        ae_score_sim = 0.0

    # ─── Métricas resultado ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Estimación Resultante (Último mes simulado)")

    ratio_sim = df_sim[DatasetKeys.CONSUMO_RATIO].iloc[-1]
    ratio_orig = df_b[DatasetKeys.CONSUMO_RATIO].iloc[-1]
    delta_ratio_pct = ((ratio_sim / ratio_orig) - 1) * 100 if ratio_orig > 0 else 0
    
    fisico_sim = consumo_fisico_sim.iloc[-1]
    fisico_orig = df_b[DatasetKeys.CONSUMO_FISICO_ESPERADO].iloc[-1] if DatasetKeys.CONSUMO_FISICO_ESPERADO in df_b.columns else fisico_sim
    delta_fisico_pct = ((fisico_sim / fisico_orig) - 1) * 100 if fisico_orig > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("💧 Ratio Consumo Simulado", f"{ratio_sim:.2f}",
              delta=f"{delta_ratio_pct:+.1f}% vs real")
    m2.metric("🟢 Esperado Físico (IA-RF)", f"{fisico_sim:.2f}",
              delta=f"{delta_fisico_pct:+.1f}% vs base",
              delta_color="inverse" if delta_fisico_pct > 10 else "normal")
    m3.metric("🚨 Riesgo Anomalía (AE-IA)", f"{ae_score_sim:.0f}%",
              delta="CRÍTICO (Anomalía)" if ae_score_sim > 100 else "NORMAL",
              delta_color="inverse" if ae_score_sim > 100 else "off")

    # ─── Curva de simulación ──────────────────────────────────────────────
    st.markdown("#### 📈 Evolución a 12 meses: Modelos Predictivos en Escenario")
    
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
    
    # Línea de Reconstrucción del Autoencoder (Modelo de IA)
    fig.add_trace(go.Scatter(
        x=fechas_str, y=consumo_ae_sim,
        mode="lines", name="Reconstrucción (LSTM-AE)",
        line=dict(color="#fca311", width=3, dash="dot"),
        line_shape='spline'
    ))
    
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
    with st.expander("🔍 Ver Importancia de Factores Físicos", expanded=False):
        importances = rf_model.feature_importances_
        feat_names_friendly = {
            DatasetKeys.TEMP_MEDIA: "Temperatura Media",
            DatasetKeys.PRECIPITACION: "Precipitación",
            DatasetKeys.PCT_VT_BARRIO_INE: "Turismo Oficial (% VT)",
            DatasetKeys.PCT_VT_SIN_REGISTRAR: "Pisos Ilegales (% VT)",
            DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA: "Plazas Hoteleras",
            DatasetKeys.NDVI_SATELITE: "Índice Vegetación (NDVI)"
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
