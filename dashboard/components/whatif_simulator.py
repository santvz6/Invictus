"""
whatif_simulator.py — Simulador What-If Interactivo
=====================================================
Permite al usuario ajustar manualmente el valor de los features
(temperatura, precipitación, NDVI, % VT sin registrar, % festivos)
para simular cómo varía el consumo mensual esperado por contrato
respecto a la línea base histórica del barrio seleccionado.

Lógica:
    consumo_simulado = fourier_base + Δ_exogeno(features)
    z_simulado       = (consumo_simulado - μ_barrio) / σ_barrio

La función principal es render_whatif(df, barrio) y se llama
desde dashboard/app.py en el tab "Simulador What-if".
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import DatasetKeys

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES / CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

# Definición de los sliders: (etiqueta, columna DatasetKey, unidad, min, max, step, ayuda)
FEATURES_WHATIF = [
    {
        "label":  "Temperatura Media (°C)",
        "col":    DatasetKeys.TEMP_MEDIA,
        "unit":   "°C",
        "min":    -5.0,
        "max":    45.0,
        "step":   0.5,
        "help":   "Temperatura media mensual estimada para el período simulado.",
        "icon":   "🌡️",
        "color":  "#f39c12",
    },
    {
        "label":  "Precipitación (mm)",
        "col":    DatasetKeys.PRECIPITACION,
        "unit":   "mm",
        "min":    0.0,
        "max":    300.0,
        "step":   1.0,
        "help":   "Total de precipitación mensual acumulada.",
        "icon":   "🌧️",
        "color":  "#4cc9f0",
    },
    {
        "label":  "NDVI Satélite (Vegetación)",
        "col":    DatasetKeys.NDVI_SATELITE,
        "unit":   "",
        "min":    -0.2,
        "max":    1.0,
        "step":   0.01,
        "help":   "Índice de vegetación normalizado (NDVI). Valores altos indican zonas verdes densas.",
        "icon":   "🌿",
        "color":  "#52b788",
    },
    {
        "label":  "% VT sin registrar",
        "col":    DatasetKeys.PCT_VT_SIN_REGISTRAR,
        "unit":   "%",
        "min":    0.0,
        "max":    100.0,
        "step":   0.5,
        "help":   "Porcentaje estimado de viviendas turísticas que operan sin estar registradas en el sector.",
        "icon":   "🏠",
        "color":  "#e74c3c",
    },
    {
        "label":  "% Días Festivos en el Mes",
        "col":    DatasetKeys.PCT_FESTIVOS,
        "unit":   "%",
        "min":    0.0,
        "max":    50.0,
        "step":   0.5,
        "help":   "Porcentaje de días del mes clasificados como festivos o puentes.",
        "icon":   "🎉",
        "color":  "#9b59b6",
    },
]

# Umbrales del semáforo (mismos que ModeloFisico._finalize_fisicos)
Z_LEVE  = 1.5
Z_MOD   = 2.0
Z_GRAVE = 2.5

COLOR_NORMAL = "#52b788"
COLOR_LEVE   = "#f39c12"
COLOR_MOD    = "#e67e22"
COLOR_GRAVE  = "#e74c3c"


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_barrio_stats(df: pd.DataFrame, barrio: str | None) -> dict:
    """
    Extrae la media y desviación estándar históricas del consumo_ratio
    para el barrio seleccionado. Si no hay barrio, usa todo el DataFrame.
    """
    if barrio:
        barrios_limpios = df[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
        df_b = df[barrios_limpios == barrio.upper()].copy()
    else:
        df_b = df.copy()

    if df_b.empty or DatasetKeys.CONSUMO_RATIO not in df_b.columns:
        return {"mu": 0.0, "sigma": 1.0, "fourier_base": 0.0, "features_mean": {}}

    mu    = df_b[DatasetKeys.CONSUMO_RATIO].mean()
    sigma = df_b[DatasetKeys.CONSUMO_RATIO].std()
    sigma = sigma if sigma > 0 else 1.0

    # Línea base Fourier: tomamos el promedio de la predicción histórica
    fourier_base = (
        df_b[DatasetKeys.CONSUMO_FISICO_ESPERADO].mean()
        if DatasetKeys.CONSUMO_FISICO_ESPERADO in df_b.columns
        else mu
    )

    # Medias históricas de cada feature del barrio (como defaults de los sliders)
    features_mean = {}
    for f in FEATURES_WHATIF:
        col = f["col"]
        if col in df_b.columns:
            features_mean[col] = float(df_b[col].mean())
        else:
            features_mean[col] = 0.0

    return {
        "mu":           mu,
        "sigma":        sigma,
        "fourier_base": fourier_base,
        "features_mean": features_mean,
    }


def _simulate_consumption(
    fourier_base: float,
    feature_values: dict,
    df_ref: pd.DataFrame,
    barrio: str | None,
) -> float:
    """
    Estima el consumo simulado usando una regresión lineal aproximada basada en
    las correlaciones históricas de cada feature con el residuo del barrio.

    consumo_sim = fourier_base + sum_i(beta_i * (x_i - mu_i))

    donde beta_i = cov(feature_i, residuo) / var(feature_i).
    """
    if barrio:
        barrios_limpios = df_ref[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
        df_b = df_ref[barrios_limpios == barrio.upper()].copy()
    else:
        df_b = df_ref.copy()

    if df_b.empty:
        return fourier_base

    # Residuo = consumo_ratio − fourier (o −prediccion fourier)
    if DatasetKeys.CONSUMO_FISICO_ESPERADO in df_b.columns:
        df_b["_residuo"] = df_b[DatasetKeys.CONSUMO_RATIO] - df_b[DatasetKeys.CONSUMO_FISICO_ESPERADO]
    elif DatasetKeys.PREDICCION_FOURIER in df_b.columns:
        df_b["_residuo"] = df_b[DatasetKeys.CONSUMO_RATIO] - df_b[DatasetKeys.PREDICCION_FOURIER]
    else:
        df_b["_residuo"] = 0.0

    delta_total = 0.0
    for f in FEATURES_WHATIF:
        col = f["col"]
        if col not in df_b.columns:
            continue

        x = df_b[col].fillna(df_b[col].mean())
        y = df_b["_residuo"].fillna(0)
        var_x = x.var()
        if var_x < 1e-9:
            continue
        beta = x.cov(y) / var_x  # OLS simple
        mu_x = x.mean()
        delta_total += beta * (feature_values.get(col, mu_x) - mu_x)

    return fourier_base + delta_total


def _get_alert_info(z: float) -> tuple[str, str, str]:
    """Devuelve (nivel_texto, color, emoji) según el z-score."""
    az = abs(z)
    sign = "EXCESO" if z > 0 else "DEFECTO"
    if az > Z_GRAVE:
        return f"{sign} GRAVE",  COLOR_GRAVE, "🔴"
    elif az > Z_MOD:
        return f"{sign} MODERADO", COLOR_MOD, "🟠"
    elif az > Z_LEVE:
        return f"{sign} LEVE", COLOR_LEVE, "🟡"
    else:
        return "NORMAL", COLOR_NORMAL, "🟢"


def _build_gauge(z_value: float, nivel: str, color: str) -> go.Figure:
    """Construye el gauge chart del Z-Score simulado."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(z_value, 3),
        number={"font": {"size": 36, "color": color}, "suffix": " σ"},
        delta={"reference": 0, "increasing": {"color": COLOR_GRAVE}, "decreasing": {"color": COLOR_NORMAL}},
        gauge={
            "axis": {"range": [-4, 4], "tickwidth": 1, "tickcolor": "#aaa", "tickfont": {"color": "#aaa"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [-4,    -Z_GRAVE], "color": "rgba(231,76,60,0.25)"},
                {"range": [-Z_GRAVE, -Z_MOD], "color": "rgba(230,126,34,0.18)"},
                {"range": [-Z_MOD,  -Z_LEVE], "color": "rgba(243,156,18,0.15)"},
                {"range": [-Z_LEVE,   Z_LEVE], "color": "rgba(82,183,136,0.15)"},
                {"range": [Z_LEVE,    Z_MOD],  "color": "rgba(243,156,18,0.15)"},
                {"range": [Z_MOD,     Z_GRAVE], "color": "rgba(230,126,34,0.18)"},
                {"range": [Z_GRAVE,    4],       "color": "rgba(231,76,60,0.25)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": z_value,
            },
        },
        title={"text": f"<b>Z-Score Simulado</b><br><span style='font-size:14px;color:{color}'>{nivel}</span>",
               "font": {"size": 14, "color": "#e0e0e0"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=10),
        height=270,
    )
    return fig


def _build_comparison_bar(consumo_base: float, consumo_sim: float, consumo_real: float) -> go.Figure:
    """Gráfico de barras comparativo: base histórica, simulación, real histórico."""
    labels = ["Base Histórica<br>(Fourier)", "Simulado<br>(What-If)", "Real Histórico<br>(Media)"]
    values = [consumo_base, consumo_sim, consumo_real]
    colors = ["#4cc9f0", "#f39c12", "#52b788"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f} m³/cto" for v in values],
        textposition="outside",
        textfont=dict(color="#e0e0e0", size=13),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="m³ / contrato", gridcolor="rgba(255,255,255,0.07)", rangemode="tozero"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        height=260,
        showlegend=False,
    )
    return fig


def _build_sensitivity_chart(
    df_ref: pd.DataFrame,
    barrio: str | None,
    fourier_base: float,
    sigma: float,
    feature_values: dict,
) -> go.Figure:
    """
    Spider/radar de sensibilidad: muestra la contribución de cada feature al
    cambio de consumo simulado respecto a la base Fourier, expresado en sigmas.
    """
    if barrio:
        barrios_limpios = df_ref[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()
        df_b = df_ref[barrios_limpios == barrio.upper()].copy()
    else:
        df_b = df_ref.copy()

    if df_b.empty:
        return go.Figure()

    if DatasetKeys.CONSUMO_FISICO_ESPERADO in df_b.columns:
        df_b["_residuo"] = df_b[DatasetKeys.CONSUMO_RATIO] - df_b[DatasetKeys.CONSUMO_FISICO_ESPERADO]
    elif DatasetKeys.PREDICCION_FOURIER in df_b.columns:
        df_b["_residuo"] = df_b[DatasetKeys.CONSUMO_RATIO] - df_b[DatasetKeys.PREDICCION_FOURIER]
    else:
        df_b["_residuo"] = 0.0

    labels = []
    contributions_z = []
    bar_colors = []

    for f in FEATURES_WHATIF:
        col = f["col"]
        if col not in df_b.columns:
            continue
        x = df_b[col].fillna(df_b[col].mean())
        y = df_b["_residuo"].fillna(0)
        var_x = x.var()
        if var_x < 1e-9:
            continue
        beta     = x.cov(y) / var_x
        mu_x     = x.mean()
        delta    = beta * (feature_values.get(col, mu_x) - mu_x)
        delta_z  = delta / sigma if sigma > 0 else 0.0
        labels.append(f["icon"] + " " + f["label"].split("(")[0].strip())
        contributions_z.append(round(delta_z, 4))
        bar_colors.append(f["color"])

    fig = go.Figure(go.Bar(
        x=labels,
        y=contributions_z,
        marker_color=bar_colors,
        text=[f"{v:+.3f}σ" for v in contributions_z],
        textposition="outside",
        textfont=dict(color="#e0e0e0", size=11),
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Δ Z-Score por feature", gridcolor="rgba(255,255,255,0.07)"),
        xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.03)"),
        height=270,
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# RENDER PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def render_whatif(df: pd.DataFrame, barrio: str | None = None) -> None:
    """
    Renderiza el simulador What-If completo en el tab de Streamlit.

    Args:
        df      : DataFrame filtrado (df_filtered de app.py).
        barrio  : Barrio activo seleccionado en el mapa (puede ser None).
    """

    # ── Encabezado ────────────────────────────────────────────────────────────
    titulo_barrio = barrio if barrio else "Todos los Barrios"
    st.markdown(
        f"""
        <div style="padding: 8px 0 18px;">
            <span style="font-size:22px; font-weight:700; color:#4cc9f0;">🧮 Simulador What-If</span>
            <span style="color:#888; font-size:14px; margin-left:12px;">
                Barrio activo: <b style="color:#f4a261;">{titulo_barrio}</b>
            </span>
        </div>
        <p style="color:#aaa; font-size:13px; margin-top:-12px; margin-bottom:18px;">
            Ajusta los sliders para simular un consumo hipotético en base al valor de los features.
            El sistema calcula el cambio esperado sobre la línea base estacional (Fourier) y lo
            expresa como Z-Score para detectar situaciones de anomalía potencial.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("No hay datos disponibles para el filtro temporal y de barrio seleccionado.")
        return

    # ── Estadísticas históricas del barrio ───────────────────────────────────
    stats = _get_barrio_stats(df, barrio)
    mu           = stats["mu"]
    sigma        = stats["sigma"]
    fourier_base = stats["fourier_base"]
    feat_means   = stats["features_mean"]

    # ── Layout: sliders izquierda | resultados derecha ────────────────────────
    col_sliders, col_results = st.columns([1, 1.3], gap="large")

    with col_sliders:
        st.markdown("#### ⚙️ Parámetros del Escenario")

        feature_values: dict[str, float] = {}
        for f in FEATURES_WHATIF:
            col     = f["col"]
            default = feat_means.get(col, (f["min"] + f["max"]) / 2)
            default = float(np.clip(default, f["min"], f["max"]))

            st.markdown(
                f"""<div style="font-size:13px; color:#ccc; margin-bottom:2px;">
                    {f["icon"]} <b>{f["label"]}</b>
                </div>""",
                unsafe_allow_html=True,
            )
            val = st.slider(
                label=f["label"],
                min_value=f["min"],
                max_value=f["max"],
                value=default,
                step=f["step"],
                help=f["help"],
                key=f"whatif_{col}",
                label_visibility="collapsed",
            )
            feature_values[col] = val

        st.markdown("---")
        # Botón reset
        if st.button("↩ Restaurar valores históricos", key="whatif_reset", type="secondary"):
            for f in FEATURES_WHATIF:
                col = f["col"]
                st.session_state[f"whatif_{col}"] = float(
                    np.clip(feat_means.get(col, (f["min"] + f["max"]) / 2), f["min"], f["max"])
                )
            st.rerun()

    # ── Simulación ────────────────────────────────────────────────────────────
    consumo_sim = _simulate_consumption(fourier_base, feature_values, df, barrio)
    z_sim = (consumo_sim - mu) / sigma
    nivel, color_nivel, emoji_nivel = _get_alert_info(z_sim)

    # ── Panel de Resultados ───────────────────────────────────────────────────
    with col_results:
        st.markdown("#### 📊 Resultado de la Simulación")

        # KPIs rápidos
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Consumo Simulado",
            f"{consumo_sim:.2f} m³/cto",
            delta=f"{consumo_sim - fourier_base:+.2f} vs base",
            delta_color="inverse",
        )
        k2.metric("Z-Score",    f"{z_sim:.3f} σ",  delta=f"{nivel}", delta_color="off")
        k3.metric("Nivel Alerta", emoji_nivel,       delta=nivel,       delta_color="off")

        # Gauge
        fig_gauge = _build_gauge(z_sim, nivel, color_nivel)
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_whatif")

        # Banner de alerta si anomalía
        if abs(z_sim) > Z_LEVE:
            st.markdown(
                f"""
                <div style="background: rgba({
                    '231,76,60' if abs(z_sim) > Z_GRAVE else
                    '230,126,34' if abs(z_sim) > Z_MOD else
                    '243,156,18'
                }, 0.12); border-left: 4px solid {color_nivel};
                padding: 12px 18px; border-radius: 6px; margin-top: 8px;">
                    <span style="color:{color_nivel}; font-weight:700; font-size:15px;">
                        {emoji_nivel} {nivel}
                    </span>
                    <span style="color:#ccc; font-size:13px; margin-left:8px;">
                        — El escenario simulado supera el umbral estadístico de anomalía (|z| &gt; {
                            Z_LEVE if abs(z_sim) <= Z_MOD else Z_MOD if abs(z_sim) <= Z_GRAVE else Z_GRAVE
                        }σ).
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(82,183,136,0.10); border-left: 4px solid {COLOR_NORMAL};
                padding: 12px 18px; border-radius: 6px; margin-top: 8px;">
                    <span style="color:{COLOR_NORMAL}; font-weight:700;">🟢 NORMAL</span>
                    <span style="color:#ccc; font-size:13px; margin-left:8px;">
                        — El escenario simulado no supera ningún umbral de anomalía.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Fila inferior: comparativa + sensibilidad ─────────────────────────────
    st.markdown("---")
    col_bar, col_sens = st.columns(2, gap="large")

    with col_bar:
        st.markdown("##### Comparativa de Consumo (m³ / contrato)")
        fig_bar = _build_comparison_bar(fourier_base, consumo_sim, mu)
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_whatif")

    with col_sens:
        st.markdown("##### Sensibilidad por Feature (Δ Z-Score)")
        fig_sens = _build_sensitivity_chart(df, barrio, fourier_base, sigma, feature_values)
        st.plotly_chart(fig_sens, use_container_width=True, key="sens_whatif")

    # ── Tabla de features ─────────────────────────────────────────────────────
    with st.expander("📋 Ver valores del escenario simulado vs. histórico"):
        rows = []
        for f in FEATURES_WHATIF:
            col   = f["col"]
            val_s = feature_values.get(col, 0.0)
            val_h = feat_means.get(col, 0.0)
            delta_pct = ((val_s - val_h) / val_h * 100) if val_h != 0 else 0.0
            rows.append({
                "Feature":           f["icon"] + " " + f["label"],
                "Histórico (media)": f"{val_h:.3f} {f['unit']}",
                "Simulado":          f"{val_s:.3f} {f['unit']}",
                "Δ vs. histórico":   f"{delta_pct:+.1f}%",
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

    # ── Nota metodológica ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="color:#555; font-size:11px; margin-top:18px; line-height:1.6;">
            <b>Nota metodológica:</b> El consumo simulado se calcula sumando a la línea base estacional (Fourier)
            el impacto marginal de cada feature estimado mediante una regresión OLS sobre el residuo histórico.
            El Z-Score se normaliza con la distribución del barrio seleccionado. Los umbrales de alerta empleados
            son: |z| &gt; 1.5 → Leve, |z| &gt; 2.0 → Moderado, |z| &gt; 2.5 → Grave.
        </div>
        """,
        unsafe_allow_html=True,
    )
