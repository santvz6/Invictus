"""
anomaly_panel.py
----------------
Panel lateral que muestra KPIs y el gráfico Real vs. Esperado
cuando el usuario selecciona un barrio en el mapa.
Panel lateral que se despliega al seleccionar un barrio en el mapa.
Muestra KPIs, gráfico comparativo Real vs Esperado y listado de anomalías.
"""

import pandas as pd
import streamlit as st
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
    # 1. Filtrar datos del barrio seleccionado
    df_b = df[df[DatasetKeys.BARRIO] == barrio].copy()
    
    st.markdown(f"### 🔎 Análisis: {barrio}")
    st.markdown("<hr style='margin: 0.5em 0; border-color: rgba(76,201,240,0.2);'/>", unsafe_allow_html=True)
    
    if df_b.empty:
        st.info("No hay datos disponibles para este barrio en el periodo seleccionado.")
        return

    df_barrio = df_barrio.sort_values(DatasetKeys.FECHA)
    df_b = df_b.sort_values(DatasetKeys.FECHA)

    # ─── KPIs principales ────────────────────────────────────────────────
    st.markdown(f"### 📍 {barrio}")
    st.markdown("---")
    # 2. KPIs Principales
    consumo_total = df_b[DatasetKeys.CONSUMO].sum()
    
    # Determinar qué columna usar como base de consumo esperado
    esperado_col = DatasetKeys.PREDICCION_FOURIER if DatasetKeys.PREDICCION_FOURIER in df_b.columns else DatasetKeys.CONSUMO_FISICO_ESPERADO
    esperado_total = df_b[esperado_col].sum() if esperado_col in df_b.columns else 0
    
    alertas = df_b["ALERTA_TURISTICA_ILEGAL"].sum() if "ALERTA_TURISTICA_ILEGAL" in df_b.columns else 0

    total_consumo  = df_barrio[DatasetKeys.CONSUMO].sum()
    avg_ratio      = df_barrio[DatasetKeys.CONSUMO_RATIO].mean() if DatasetKeys.CONSUMO_RATIO in df_barrio else None
    pct_vt         = df_barrio[DatasetKeys.PCT_VT_BARRIO].mean() if DatasetKeys.PCT_VT_BARRIO in df_barrio else None
    num_alertas    = int(df_barrio["ALERTA_TURISTICA_ILEGAL"].sum()) if "ALERTA_TURISTICA_ILEGAL" in df_barrio else 0
    avg_error      = df_barrio["reconstruction_error"].mean() if "reconstruction_error" in df_barrio else None
    c1, c2 = st.columns(2)
    c1.metric("Consumo Total", f"{consumo_total:,.0f} m³", 
              delta=f"{(consumo_total - esperado_total):+,.0f} m³ vs Esperado", delta_color="inverse")
    c2.metric("Alertas Detectadas", int(alertas), 
              delta="CRÍTICO" if alertas > 0 else "Normal", delta_color="inverse" if alertas > 0 else "normal")

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
    # 3. Gráfico Comparativo: Consumo Real vs Esperado
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
    # Línea de Consumo Esperado
    if esperado_col in df_b.columns:
        fig.add_trace(go.Scatter(
            x=df_b[DatasetKeys.FECHA], 
            y=df_b[esperado_col],
            mode='lines', 
            name='Consumo Esperado',
            line=dict(color='#52b788', width=2, dash='dash')
        ))

    # Consumo esperado (modelo físico)
    # Línea de Consumo Real
    fig.add_trace(go.Scatter(
        x=fechas, y=esperado,
        mode="lines",
        line=dict(color="#52b788", width=2, dash="dash"),
        name="Consumo Esperado (física)",
        x=df_b[DatasetKeys.FECHA], 
        y=df_b[DatasetKeys.CONSUMO],
        mode='lines+markers', 
        name='Consumo Real',
        line=dict(color='#4cc9f0', width=2),
        marker=dict(size=4)
    ))

    # Consumo real
    fig.add_trace(go.Scatter(
        x=fechas, y=real,
        mode="lines+markers",
        line=dict(color="#4cc9f0", width=2),
        marker=dict(size=4),
        name="Consumo Real",
    ))
    # Resaltar puntos exactos de Anomalía
    if "ALERTA_TURISTICA_ILEGAL" in df_b.columns:
        df_anomalias = df_b[df_b["ALERTA_TURISTICA_ILEGAL"] == True]
        if not df_anomalias.empty:
            fig.add_trace(go.Scatter(
                x=df_anomalias[DatasetKeys.FECHA], 
                y=df_anomalias[DatasetKeys.CONSUMO],
                mode='markers', 
                name='Anomalía Detectada',
                marker=dict(color='#ff4b4b', size=10, symbol='x', line=dict(width=2, color='white')),
                hovertext=df_anomalias["reconstruction_error"].apply(lambda x: f"Error AE: {x:.3f}"),
                hoverinfo="text+x+y"
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
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="", gridcolor="#2a3a4a"),
        yaxis=dict(title="m³", gridcolor="#2a3a4a"),
        height=300,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Metros Cúbicos (m³)", title_font=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        height=280
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Listado Tabular de Anomalías
    st.markdown("#### 🚨 Registro de Anomalías")
    if "ALERTA_TURISTICA_ILEGAL" in df_b.columns and alertas > 0:
        cols_to_show = [DatasetKeys.FECHA, DatasetKeys.CONSUMO, esperado_col, "reconstruction_error"]
        cols_to_show = [c for c in cols_to_show if c in df_b.columns]
        
        df_table = df_anomalias[cols_to_show].copy()
        df_table[DatasetKeys.FECHA] = df_table[DatasetKeys.FECHA].dt.strftime("%Y-%m")
        
        # Formateo y renombrado visual
        df_table = df_table.rename(columns={
            DatasetKeys.FECHA: "Mes",
            DatasetKeys.CONSUMO: "Real (m³)",
            esperado_col: "Esperado (m³)",
            "reconstruction_error": "Error Autoencoder"
        })
        
        st.dataframe(df_table.style.format(precision=1), hide_index=True, use_container_width=True)
    else:
        st.success("✅ Sin comportamiento anómalo detectado en este periodo.")
