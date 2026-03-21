"""
anomaly_panel.py
----------------
Panel lateral que muestra KPIs y el gráfico Real vs. Esperado
cuando el usuario selecciona un barrio en el mapa.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import DatasetKeys


def render_anomaly_panel(df: pd.DataFrame, barrio: str):
    """
    Renderiza el sidebar con información detallada del barrio seleccionado.

    Parameters
    ----------
    df : pd.DataFrame  — DataFrame completo (serie temporal, sin agregar)
    barrio : str       — Nombre del barrio seleccionado
    """
    df_barrio = df[df[DatasetKeys.BARRIO] == barrio].copy()

    if df_barrio.empty:
        st.warning(f"No hay datos para el barrio **{barrio}**")
        return

    df_barrio = df_barrio.sort_values(DatasetKeys.FECHA)

    # ─── KPIs principales ────────────────────────────────────────────────
    st.markdown(f"### 📍 {barrio}")
    st.markdown("---")

    total_consumo  = df_barrio[DatasetKeys.CONSUMO].sum()
    avg_ratio      = df_barrio[DatasetKeys.CONSUMO_RATIO].mean() if DatasetKeys.CONSUMO_RATIO in df_barrio else None
    pct_vt         = df_barrio[DatasetKeys.PCT_VT_BARRIO].mean() if DatasetKeys.PCT_VT_BARRIO in df_barrio else None
    num_alertas    = int(df_barrio["ALERTA_TURISTICA_ILEGAL"].sum()) if "ALERTA_TURISTICA_ILEGAL" in df_barrio else 0
    avg_error      = df_barrio["reconstruction_error"].mean() if "reconstruction_error" in df_barrio else None

    col1, col2 = st.columns(2)
    col1.metric("💧 Consumo Total", f"{total_consumo:,.0f} m³")
    col2.metric("🚨 Alertas detectadas", num_alertas,
                delta="⚠️ ALTO" if num_alertas > 3 else None,
                delta_color="inverse")

    col3, col4 = st.columns(2)
    if avg_ratio is not None:
        col3.metric("📊 Ratio Medio", f"{avg_ratio:.2f}")
    if pct_vt is not None:
        col4.metric("🏖 % VT", f"{pct_vt:.1f}%")
    if avg_error is not None:
        st.metric("🤖 Error Reconstrucción (LSTM-AE)", f"{avg_error:.4f}",
                  delta="Alto" if avg_error > 0.15 else "Normal",
                  delta_color="inverse" if avg_error > 0.15 else "off")

    st.markdown("---")

    # ─── Lista de anomalías ───────────────────────────────────────────────
    if "ALERTA_TURISTICA_ILEGAL" in df_barrio.columns:
        df_anomalias = df_barrio[df_barrio["ALERTA_TURISTICA_ILEGAL"] == True]
        if not df_anomalias.empty:
            st.markdown("#### 🔴 Periodos con Alerta")
            for _, row in df_anomalias.iterrows():
                fecha_str = row[DatasetKeys.FECHA].strftime("%b %Y") if hasattr(row[DatasetKeys.FECHA], "strftime") else str(row[DatasetKeys.FECHA])
                error_val = row.get("reconstruction_error", 0)
                st.markdown(
                    f"""<div style="background:rgba(193,18,31,0.15); border-left:3px solid #c1121f;
                    padding:6px 10px; border-radius:4px; margin:4px 0; font-size:12px;">
                    📅 <b>{fecha_str}</b> — Error: <b>{error_val:.3f}</b>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ Sin alertas en el periodo seleccionado")

    st.markdown("---")

    # ─── Gráfico Real vs. Esperado ────────────────────────────────────────
    st.markdown("#### 📈 Consumo Real vs. Esperado")

    if DatasetKeys.CONSUMO_FISICO_ESPERADO not in df_barrio.columns:
        st.info("Columna de consumo esperado no disponible.")
        return

    fechas = df_barrio[DatasetKeys.FECHA]
    real   = df_barrio[DatasetKeys.CONSUMO]
    esperado = df_barrio[DatasetKeys.CONSUMO_FISICO_ESPERADO]

    # Marcadores de anomalía
    if "ALERTA_TURISTICA_ILEGAL" in df_barrio.columns:
        mask_anom = df_barrio["ALERTA_TURISTICA_ILEGAL"] == True
        anom_x = fechas[mask_anom]
        anom_y = real[mask_anom]
    else:
        anom_x = pd.Series([], dtype="datetime64[ns]")
        anom_y = pd.Series([], dtype=float)

    fig = go.Figure()

    # Área rellena "umbral físico" (1.5x esperado)
    fig.add_trace(go.Scatter(
        x=pd.concat([fechas, fechas[::-1]]),
        y=pd.concat([esperado * 1.5, esperado[::-1]]),
        fill="toself",
        fillcolor="rgba(193,18,31,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Zona Sospechosa (>1.5x esperado)",
        showlegend=True,
    ))

    # Consumo esperado (modelo físico)
    fig.add_trace(go.Scatter(
        x=fechas, y=esperado,
        mode="lines",
        line=dict(color="#52b788", width=2, dash="dash"),
        name="Consumo Esperado (física)",
    ))

    # Consumo real
    fig.add_trace(go.Scatter(
        x=fechas, y=real,
        mode="lines+markers",
        line=dict(color="#4cc9f0", width=2),
        marker=dict(size=4),
        name="Consumo Real",
    ))

    # Puntos de anomalía
    if not anom_x.empty:
        fig.add_trace(go.Scatter(
            x=anom_x, y=anom_y,
            mode="markers",
            marker=dict(color="#c1121f", size=10, symbol="star",
                        line=dict(color="white", width=1)),
            name="🚨 Anomalía detectada",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="", gridcolor="#2a3a4a"),
        yaxis=dict(title="m³", gridcolor="#2a3a4a"),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
