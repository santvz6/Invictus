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

from src.config import DatasetKeys


def render_whatif(df: pd.DataFrame, barrio: str | None = None):
    """
    Renderiza el panel simulador.

    Parameters
    ----------
    df : pd.DataFrame  — DataFrame completo para calcular rangos reales
    barrio : str | None — Si se pasa, muestra los valores reales como referencia
    """
    st.markdown("### 🔬 Simulador What-if")
    st.markdown(
        "<small style='color:#888;'>Ajusta las variables para ver cómo cambia el consumo estimado "
        "y la probabilidad de anomalía.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ─── Rangos reales del dataset ───────────────────────────────────────
    def _safe_range(col, default_lo, default_hi):
        if col in df.columns:
            lo, hi = float(df[col].min()), float(df[col].max())
            return (lo, hi) if lo < hi else (default_lo, default_hi)
        return (default_lo, default_hi)

    temp_range    = _safe_range(DatasetKeys.TEMP_MEDIA,      5.0,  40.0)
    precip_range  = _safe_range(DatasetKeys.PRECIPITACION,   0.0, 120.0)
    vt_range      = _safe_range(DatasetKeys.NUM_VT_BARRIO_INE,   0.0, 200.0)
    ratio_range   = _safe_range(DatasetKeys.CONSUMO_RATIO,   0.5,  30.0)

    # Valores por defecto: mediana del barrio si está disponible
    defaults: dict = {}
    if barrio:
        df_b = df[df[DatasetKeys.BARRIO] == barrio]
        for col, key in [(DatasetKeys.TEMP_MEDIA, "temp"),
                         (DatasetKeys.PRECIPITACION, "precip"),
                         (DatasetKeys.NUM_VT_BARRIO_INE, "vt"),
                         (DatasetKeys.CONSUMO_RATIO, "ratio")]:
            if col in df_b.columns and not df_b[col].empty:
                defaults[key] = float(df_b[col].median())

    def_temp   = defaults.get("temp",   np.mean(temp_range))
    def_precip = defaults.get("precip", np.mean(precip_range))
    def_vt     = defaults.get("vt",     np.mean(vt_range))
    def_ratio  = defaults.get("ratio",  np.mean(ratio_range))

    # ─── Sliders ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        temp_val = st.slider(
            "🌡 Temperatura Media (°C)",
            min_value=temp_range[0], max_value=temp_range[1],
            value=float(def_temp), step=0.5, key="wif_temp",
        )
        vt_val = st.slider(
            "🏖 Nº Viviendas Turísticas",
            min_value=int(vt_range[0]), max_value=int(vt_range[1]),
            value=int(def_vt), step=1, key="wif_vt",
        )

    with col2:
        precip_val = st.slider(
            "🌧 Precipitación (mm)",
            min_value=precip_range[0], max_value=precip_range[1],
            value=float(def_precip), step=1.0, key="wif_precip",
        )
        ratio_val = st.slider(
            "📊 Ratio Consumo/Contrato",
            min_value=ratio_range[0], max_value=ratio_range[1],
            value=float(def_ratio), step=0.1, key="wif_ratio",
        )

    # ─── Modelo simplificado de estimación ───────────────────────────────
    # Referencia: mediana del barrio o global
    if barrio and barrio in df[DatasetKeys.BARRIO].values:
        df_ref = df[df[DatasetKeys.BARRIO] == barrio]
    else:
        df_ref = df

    base_consumo_esperado = float(df_ref[DatasetKeys.CONSUMO_FISICO_ESPERADO].median()) \
        if DatasetKeys.CONSUMO_FISICO_ESPERADO in df_ref.columns else 5000.0

    # Factores de influencia (lineales simplificados, calibrados con el proyecto)
    delta_temp    = (temp_val - def_temp) * 0.03    # +3% por cada grado
    delta_precip  = (precip_val - def_precip) * (-0.005)  # -0.5% por mm de lluvia
    delta_vt      = (vt_val - def_vt) * 0.008      # +0.8% por cada VT extra
    delta_ratio   = (ratio_val - def_ratio) * 0.05  # +5% por unidad de ratio

    factor_total = 1 + delta_temp + delta_precip + delta_vt + delta_ratio
    consumo_simulado = base_consumo_esperado * factor_total

    # Probabilidad de anomalía (heurística)
    ratio_real_esperado = (ratio_val * 100) / base_consumo_esperado
    prob_anomalia = min(1.0, max(0.0, 0.05 + delta_vt * 2 + delta_ratio * 0.5 + (0.1 if factor_total > 1.5 else 0)))

    # ─── Métricas resultado ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Estimación Resultante")

    m1, m2, m3 = st.columns(3)
    m1.metric("💧 Consumo Simulado (m³)", f"{consumo_simulado:,.0f}",
              delta=f"{(factor_total - 1)*100:+.1f}% vs. base")
    m2.metric("⚡ Variación Total", f"{(factor_total - 1)*100:+.1f}%",
              delta_color="inverse" if factor_total > 1.3 else "normal")
    m3.metric("🚨 Riesgo Anomalía", f"{prob_anomalia*100:.0f}%",
              delta="ALTO" if prob_anomalia > 0.4 else "BAJO",
              delta_color="inverse" if prob_anomalia > 0.4 else "off")

    # ─── Gráfico tornado de contribuciones ───────────────────────────────
    contribuciones = {
        "Temperatura":        delta_temp * 100,
        "Precipitación":      delta_precip * 100,
        "VT (turismo)":       delta_vt * 100,
        "Ratio consumo":      delta_ratio * 100,
    }

    colors = ["#c1121f" if v > 0 else "#52b788" for v in contribuciones.values()]
    fig = go.Figure(go.Bar(
        x=list(contribuciones.values()),
        y=list(contribuciones.keys()),
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in contribuciones.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Contribución de cada factor al consumo (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.6)",
        margin=dict(l=0, r=10, t=40, b=0),
        xaxis=dict(title="%", gridcolor="#2a3a4a", zeroline=True, zerolinecolor="#555"),
        yaxis=dict(gridcolor="#2a3a4a"),
        height=250,
    )
    st.plotly_chart(fig, width='stretch')

    # ─── Curva de simulación ──────────────────────────────────────────────
    st.markdown("#### 🌡 Curva: Temperatura vs. Consumo (todo lo demás fijo)")
    temps = np.linspace(temp_range[0], temp_range[1], 60)
    consumos_curva = [
        base_consumo_esperado * (1 + (t - def_temp) * 0.03 + delta_precip + delta_vt + delta_ratio)
        for t in temps
    ]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=temps, y=consumos_curva,
        mode="lines", line=dict(color="#4cc9f0", width=2),
        name="Consumo estimado",
    ))
    fig2.add_vline(x=temp_val, line_dash="dash", line_color="#f4a261",
                   annotation_text=f"Actual: {temp_val}°C", annotation_position="top right")
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Temperatura (°C)", gridcolor="#2a3a4a"),
        yaxis=dict(title="m³ estimados", gridcolor="#2a3a4a"),
        height=220,
    )
    st.plotly_chart(fig2, width='stretch')
