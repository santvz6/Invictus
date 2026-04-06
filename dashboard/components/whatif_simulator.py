"""
whatif_simulator.py — Simulador What-If Interactivo v2
=======================================================
Mejoras respecto a v1:

  1. BETAS NO-LINEALES POR CUANTIL
     En lugar de un único coeficiente OLS (Pearson×σ), se calcula la pendiente
     local en 3 tramos (bajo/medio/alto) via regresión por cuantil simplificada.
     El slider elige automáticamente el tramo correcto del barrio seleccionado.

  2. MODO ESTACIONAL (MES DE SIMULACIÓN)
     El usuario puede anclar la simulación a un mes concreto del año.
     El delta_total se escala por el ratio estacional Fourier del barrio en ese mes,
     capturando que la misma desviación de temperatura impacta diferente en enero vs. julio.

  3. SCORE DE PLAUSIBILIDAD (Mahalanobis)
     Calcula cuán probable es la combinación de features propuesta dado el histórico
     del barrio. Se muestra como "Escenario Plausible / Poco Común / Atípico" con
     el percentil de esa distancia en la distribución empírica.

  4. RADAR CHART DE DESVIACIONES
     Gráfico tipo araña que muestra las 5 features como % de desviación respecto a
     su media histórica, dando una lectura global del escenario de un vistazo.

  5. CACHÉ DE BETAS EN SESSION_STATE
     Los betas se recalculan solo cuando cambia el barrio, no en cada re-render
     por movimiento de slider, reduciendo latencia significativamente.

  6. PERFIL ANUAL SIMULADO vs REAL
     La fila inferior muestra los 12 meses históricos del barrio con el mes
     simulado destacado, contextualizando el escenario en la estacionalidad real.

Lógica de simulación:
    beta_i(x)   = pendiente local en el tramo de x dentro de la distribución histórica
    delta_i     = beta_i(val_sim) * (val_sim - mu_i_tramo)
    delta_total = Σ delta_i  [en m³/cto]
    delta_scaled= delta_total × factor_estacional(mes)
    consumo_sim = fourier_mes + delta_scaled
    z_sim       = delta_scaled / sigma_residuo

Anclaje: z = 0 cuando TODOS los sliders están en sus medias históricas (delta_i = 0).
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import mahalanobis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import DatasetKeys

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES / CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

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
        "label":  "Pernoctaciones Turísticas",
        "col":    DatasetKeys.PERNOCT_VT_PROV_INE,
        "unit":   "noches/mes",
        "min":    0.0,
        "max":    200000.0,
        "step":   2500.0,
        "help":   "Noches ocupadas mensuales (Pernoctaciones) estimadas (INE).",
        "icon":   "🏖️",
        "color":  "#e74c3c",
    },
    {
        "label":  "Días Festivos en el Mes",
        "col":    DatasetKeys.DIAS_FESTIVOS,
        "unit":   "días",
        "min":    0.0,
        "max":    12.0,
        "step":   1.0,
        "help":   "Número total de días festivos que caen en el mes simulado.",
        "icon":   "🎉",
        "color":  "#9b59b6",
    },
]

# Nombres de los meses en castellano
MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

# Umbrales del semáforo (mismos que ModeloFisico._finalize_fisicos)
Z_LEVE  = 1.5
Z_MOD   = 2.0
Z_GRAVE = 2.5

COLOR_NORMAL = "#52b788"
COLOR_LEVE   = "#f39c12"
COLOR_MOD    = "#e67e22"
COLOR_GRAVE  = "#e74c3c"


# ──────────────────────────────────────────────────────────────────────────────
# MOTOR DE CÁLCULO (WhatIfEngine) — separado del renderizado
# ──────────────────────────────────────────────────────────────────────────────

class WhatIfEngine:
    """
    Encapsula todos los cálculos del simulador.
    Se instancia una vez por barrio y se cachea en session_state.
    """

    def __init__(self, df_barrio: pd.DataFrame):
        """
        Prepara las estadísticas históricas y los betas no-lineales.
        df_barrio debe estar ya filtrado al barrio activo.
        """
        self.df = df_barrio.copy()
        self._prepare()

    def _prepare(self):
        df = self.df

        # ── Consumo ratio y línea base ────────────────────────────────────────
        self.mu    = float(df[DatasetKeys.CONSUMO_RATIO].mean()) if DatasetKeys.CONSUMO_RATIO in df.columns else 0.0
        self.sigma = float(df[DatasetKeys.CONSUMO_RATIO].std())
        self.sigma = self.sigma if self.sigma > 0 else 1.0

        self.fourier_base = (
            float(df[DatasetKeys.CONSUMO_FISICO_ESPERADO].mean())
            if DatasetKeys.CONSUMO_FISICO_ESPERADO in df.columns
            else self.mu
        )

        # ── Residuo sobre el que se estiman los efectos ──────────────────────
        if DatasetKeys.CONSUMO_FISICO_ESPERADO in df.columns:
            df["_residuo"] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.CONSUMO_FISICO_ESPERADO]
        elif DatasetKeys.PREDICCION_FOURIER in df.columns:
            df["_residuo"] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.PREDICCION_FOURIER]
        else:
            df["_residuo"] = 0.0

        self.sigma_residuo = float(df["_residuo"].std())
        if self.sigma_residuo <= 0 or np.isnan(self.sigma_residuo):
            self.sigma_residuo = self.sigma

        # ── Medias y std de cada feature ─────────────────────────────────────
        self.feat_stats: dict[str, dict] = {}
        if DatasetKeys.FECHA in df.columns:
            df["_mes_m"] = pd.to_datetime(df[DatasetKeys.FECHA], errors="coerce").dt.month
        else:
            df["_mes_m"] = 1

        for f in FEATURES_WHATIF:
            col = f["col"]
            if col not in df.columns:
                continue
            x = df[col].dropna()
            if len(x) < 3:
                continue
                
            m_means = df.groupby("_mes_m")[col].mean().to_dict()
            
            self.feat_stats[col] = {
                "mean": float(x.mean()),
                "monthly_means": m_means,
                "std":  float(x.std()) if float(x.std()) > 0 else 1.0,
                "q33":  float(x.quantile(0.33)),
                "q66":  float(x.quantile(0.66)),
            }

        # ── Betas no-lineales por tramo (bajo/medio/alto) ────────────────────
        self.betas: dict[str, dict[str, float]] = {}
        self._compute_nonlinear_betas(df)

        # ── Perfil mensual Fourier (media por mes del año) ────────────────────
        self.fourier_monthly: dict[int, float] = {}
        self._compute_fourier_monthly(df)

        # ── Matriz de covarianza para Mahalanobis ────────────────────────────
        self._prepare_mahalanobis(df)

    def _compute_nonlinear_betas(self, df: pd.DataFrame):
        """
        Para cada feature, calcula tres betas (bajo/medio/alto) dividiendo el histórico
        en tres terciles y haciendo regresión lineal dentro de cada tramo.
        Esto captura relaciones tipo: temperatura baja → poco impacto,
        temperatura alta → gran impacto (umbral de calor).
        """
        y = df["_residuo"].fillna(0).values

        for f in FEATURES_WHATIF:
            col = f["col"]
            if col not in self.feat_stats:
                continue

            x = df[col].fillna(df[col].mean()).values
            q33 = self.feat_stats[col]["q33"]
            q66 = self.feat_stats[col]["q66"]

            betas_col = {}
            for tramo, mask in [
                ("bajo",  x <= q33),
                ("medio", (x > q33) & (x <= q66)),
                ("alto",  x > q66),
            ]:
                x_t = x[mask]
                y_t = y[mask]
                if len(x_t) < 3:
                    # Fallback: beta global
                    std_x = self.feat_stats[col]["std"]
                    std_y = float(np.std(y)) if np.std(y) > 0 else 1.0
                    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
                    r = 0.0 if np.isnan(r) else r
                    betas_col[tramo] = r * std_y / std_x
                else:
                    # OLS local: β = Cov(x,y)/Var(x)
                    cov = float(np.cov(x_t, y_t)[0, 1])
                    var = float(np.var(x_t))
                    betas_col[tramo] = cov / var if var > 1e-9 else 0.0

            self.betas[col] = betas_col

    def _compute_fourier_monthly(self, df: pd.DataFrame):
        """Calcula la media del consumo_fisico_esperado por mes del año (1-12)."""
        col_base = (
            DatasetKeys.CONSUMO_FISICO_ESPERADO
            if DatasetKeys.CONSUMO_FISICO_ESPERADO in df.columns
            else DatasetKeys.PREDICCION_FOURIER
            if DatasetKeys.PREDICCION_FOURIER in df.columns
            else None
        )
        if col_base and DatasetKeys.FECHA in df.columns:
            df_m = df.copy()
            df_m["_mes"] = pd.to_datetime(df_m[DatasetKeys.FECHA]).dt.month
            for mes in range(1, 13):
                vals = df_m.loc[df_m["_mes"] == mes, col_base]
                self.fourier_monthly[mes] = float(vals.mean()) if len(vals) > 0 else self.fourier_base
        else:
            for mes in range(1, 13):
                self.fourier_monthly[mes] = self.fourier_base

    def _prepare_mahalanobis(self, df: pd.DataFrame):
        """Prepara la matriz de covarianza inversa para el cálculo de distancia de Mahalanobis."""
        cols = [f["col"] for f in FEATURES_WHATIF if f["col"] in df.columns]
        if len(cols) < 2:
            self._mah_cols = []
            self._mah_VI = None
            self._mah_mu = None
            return

        X = df[cols].dropna()
        if len(X) < len(cols) + 2:
            self._mah_cols = []
            self._mah_VI = None
            self._mah_mu = None
            return

        self._mah_cols = cols
        self._mah_mu = X.mean().values
        try:
            cov = np.cov(X.values.T)
            self._mah_VI = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
        except np.linalg.LinAlgError:
            self._mah_VI = None
            self._mah_mu = None

        # Distribución empírica de distancias (para saber en qué percentil cae el escenario)
        if self._mah_VI is not None:
            dists = [mahalanobis(row, self._mah_mu, self._mah_VI) for row in X.values]
            self._mah_dist_pct = np.percentile(dists, [50, 75, 90, 95, 99])
        else:
            self._mah_dist_pct = None

    # ── API pública ───────────────────────────────────────────────────────────

    def get_feat_means(self, mes: int | None = None) -> dict[str, float]:
        res = {}
        for col, stats in self.feat_stats.items():
            if mes is not None and "monthly_means" in stats and mes in stats["monthly_means"]:
                res[col] = float(stats["monthly_means"][mes])
            else:
                res[col] = float(stats["mean"])
        return res

    def get_feat_stats(self) -> dict[str, dict]:
        return self.feat_stats

    def simulate(
        self,
        feature_values: dict[str, float],
        mes_simulacion: int | None = None,
    ) -> dict:
        """
        Simula el consumo para el vector de features dado.

        Args:
            feature_values: {col: valor_slider}
            mes_simulacion: mes del año (1-12) o None para usar la media anual

        Returns:
            dict con consumo_sim, z_sim, delta_total, delta_por_feature,
                  fourier_base_mes, factor_estacional, plausibilidad
        """
        # Línea base estacional para el mes elegido
        if mes_simulacion is not None:
            fourier_mes = self.fourier_monthly.get(mes_simulacion, self.fourier_base)
        else:
            fourier_mes = self.fourier_base

        # Factor de estacionalidad: ratio entre la línea base del mes y la media anual
        if self.fourier_base > 0:
            factor_estacional = fourier_mes / self.fourier_base
        else:
            factor_estacional = 1.0
        factor_estacional = float(np.clip(factor_estacional, 0.3, 3.0))

        # ── Δ por feature con beta no-lineal ─────────────────────────────────
        delta_por_feature: dict[str, float] = {}
        delta_total = 0.0

        for f in FEATURES_WHATIF:
            col = f["col"]
            if col not in self.betas:
                continue

            stats = self.feat_stats.get(col, {})
            q33 = stats.get("q33", stats.get("mean", 0))
            q66 = stats.get("q66", stats.get("mean", 0))
            
            # ── Centrado dinámico según mes para el cálculo de impacto ────────
            if mes_simulacion is not None and "monthly_means" in stats and mes_simulacion in stats["monthly_means"]:
                mu_x = float(stats["monthly_means"][mes_simulacion])
            else:
                mu_x = float(stats.get("mean", 0.0))

            val = feature_values.get(col, mu_x)

            # Elegir el tramo beta en función del valor simulado
            if val <= q33:
                tramo = "bajo"
            elif val <= q66:
                tramo = "medio"
            else:
                tramo = "alto"

            beta = self.betas[col].get(tramo, 0.0)

            # Usamos siempre la media histórica como ancla del slider para reversibilidad perfecta
            desviacion = val - mu_x
            contribucion = beta * desviacion

            # Cap individual: máx ±1.2 * sigma_residuo para que el escenario extremo llegue a Grave
            cap = 1.2 * self.sigma_residuo
            contribucion = float(np.clip(contribucion, -cap, cap))

            delta_por_feature[col] = contribucion
            delta_total += contribucion

        # Escalar el delta por la estacionalidad del mes
        delta_scaled = delta_total * factor_estacional
        consumo_sim  = fourier_mes + delta_scaled

        z_sim = delta_scaled / self.sigma_residuo if self.sigma_residuo > 0 else 0.0

        # ── Score de plausibilidad (Mahalanobis) ─────────────────────────────
        plausibilidad = self._score_plausibilidad(feature_values, mes_simulacion)

        return {
            "consumo_sim":       consumo_sim,
            "z_sim":             z_sim,
            "delta_total":       delta_total,
            "delta_scaled":      delta_scaled,
            "delta_por_feature": delta_por_feature,
            "fourier_mes":       fourier_mes,
            "factor_estacional": factor_estacional,
            "plausibilidad":     plausibilidad,
        }

    def _score_plausibilidad(self, feature_values: dict[str, float], mes_simulacion: int | None = None) -> dict:
        """Retorna nivel de plausibilidad evaluando la anomalía transpuesta a la distribución global."""
        if not self._mah_cols or self._mah_VI is None:
            return {"nivel": "sin_datos", "percentil": None, "distancia": None}

        punto_eval = []
        for i, c in enumerate(self._mah_cols):
            val_sim = feature_values.get(c, self._mah_mu[i])
            if mes_simulacion is not None and c in self.feat_stats and "monthly_means" in self.feat_stats[c]:
                mu_x_mes = float(self.feat_stats[c]["monthly_means"].get(mes_simulacion, self._mah_mu[i]))
            else:
                mu_x_mes = float(self.feat_stats.get(c, {}).get("mean", self._mah_mu[i]))
            
            # Traslación inteligente: Calculamos la anomalía respecto a la normalidad del mes,
            # y se la aplicamos a la media anual. Evaluamos la excentricidad de la *anomalía*, no del valor bruto.
            anomalia = val_sim - mu_x_mes
            val_eval = self._mah_mu[i] + anomalia
            punto_eval.append(val_eval)

        punto = np.array(punto_eval)
        dist = float(mahalanobis(punto, self._mah_mu, self._mah_VI))

        if self._mah_dist_pct is not None:
            if dist <= self._mah_dist_pct[1]:    # ≤ p75
                nivel, pct = "plausible", 75
            elif dist <= self._mah_dist_pct[2]:  # ≤ p90
                nivel, pct = "poco_comun", 90
            elif dist <= self._mah_dist_pct[3]:  # ≤ p95
                nivel, pct = "atipico", 95
            else:
                nivel, pct = "extremo", 99
        else:
            nivel, pct = "sin_datos", None

        return {"nivel": nivel, "percentil": pct, "distancia": dist}

    def get_annual_profile(self) -> pd.DataFrame:
        """Devuelve el perfil mensual histórico (consumo_ratio medio por mes)."""
        if DatasetKeys.FECHA not in self.df.columns or DatasetKeys.CONSUMO_RATIO not in self.df.columns:
            return pd.DataFrame()
        df_m = self.df.copy()
        df_m["_mes"] = pd.to_datetime(df_m[DatasetKeys.FECHA]).dt.month
        profile = df_m.groupby("_mes")[DatasetKeys.CONSUMO_RATIO].mean().reset_index()
        profile.columns = ["mes", "consumo_ratio_medio"]
        # Añadir línea Fourier
        profile["fourier"] = profile["mes"].map(self.fourier_monthly)
        return profile


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE UI
# ──────────────────────────────────────────────────────────────────────────────

def _get_engine(df: pd.DataFrame, barrio: str | None) -> WhatIfEngine:
    """
    Devuelve el WhatIfEngine del barrio activo, recalculándolo solo si el barrio cambia.
    Usa session_state para cachear el objeto entre re-renders.
    """
    cache_key = f"_whatif_engine_{barrio}"

    if cache_key not in st.session_state:
        # Filtrar al barrio
        if barrio:
            mask = df[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper() == barrio.upper()
            df_b = df[mask].copy()
        else:
            df_b = df.copy()

        if df_b.empty:
            df_b = df.copy()

        st.session_state[cache_key] = WhatIfEngine(df_b)

    return st.session_state[cache_key]


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


def _build_radar_chart(engine: WhatIfEngine, feature_values: dict[str, float]) -> go.Figure:
    """
    Radar (spider) chart que muestra la desviación % de cada feature
    respecto a su media histórica del barrio.
    Punto 0 = valores históricos, exterior = desviación máxima.
    """
    feats_disponibles = [f for f in FEATURES_WHATIF if f["col"] in engine.feat_stats]
    if not feats_disponibles:
        return go.Figure()

    labels = [f["icon"] + " " + f["label"].split("(")[0].strip() for f in feats_disponibles]
    colors_f = [f["color"] for f in feats_disponibles]

    # Desviación en % respecto a la media histórica (+100% = doble de la media)
    desvs = []
    for f in feats_disponibles:
        col = f["col"]
        mu_x  = engine.feat_stats[col]["mean"]
        std_x = engine.feat_stats[col]["std"]
        val   = feature_values.get(col, mu_x)
        # Usamos nσ como escala para que sea adimensional y comparable entre features
        d_sigma = (val - mu_x) / std_x if std_x > 0 else 0.0
        # Clampeamos a [-3, 3] para que el radar sea legible
        desvs.append(float(np.clip(d_sigma, -3, 3)))

    # Cerramos el polígono
    labels_closed = labels + [labels[0]]
    desvs_closed  = desvs  + [desvs[0]]

    fig = go.Figure()

    # Zona de normalidad (±1σ)
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * (len(labels) + 1),
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(82,183,136,0.08)",
        line=dict(color="rgba(82,183,136,0.3)", dash="dot"),
        name="±1σ Normal",
    ))

    # Zona de alerta (±2σ)
    fig.add_trace(go.Scatterpolar(
        r=[2.0] * (len(labels) + 1),
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(243,156,18,0.05)",
        line=dict(color="rgba(243,156,18,0.2)", dash="dot"),
        name="±2σ Alerta",
    ))

    # Escenario simulado (usando |desv| para el radio, indicando dirección con dash)
    r_sim = [abs(d) for d in desvs_closed]
    fig.add_trace(go.Scatterpolar(
        r=r_sim,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(76,201,240,0.15)",
        line=dict(color="#4cc9f0", width=2),
        name="Escenario",
        hovertemplate="%{theta}<br>Desviación: %{r:.2f}σ<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 3],
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#888", size=9),
                ticksuffix="σ",
            ),
            angularaxis=dict(
                tickfont=dict(color="#ccc", size=10),
                gridcolor="rgba(255,255,255,0.1)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
        showlegend=False,
        template="plotly_dark",
    )
    return fig


def _build_annual_profile(
    engine: WhatIfEngine,
    consumo_sim: float,
    mes_simulacion: int | None,
) -> go.Figure:
    """
    Muestra el perfil anual histórico del barrio (consumo_ratio medio por mes)
    con el punto simulado destacado en el mes seleccionado.
    """
    profile = engine.get_annual_profile()
    if profile.empty:
        return go.Figure()

    meses_labels = [MESES_ES.get(m, str(m))[:3] for m in profile["mes"]]

    fig = go.Figure()

    # Banda de normalidad ±1.5σ
    sigma_m = engine.sigma
    fig.add_trace(go.Scatter(
        x=meses_labels + meses_labels[::-1],
        y=(profile["fourier"] + 1.5 * sigma_m).tolist() + (profile["fourier"] - 1.5 * sigma_m).clip(lower=0).tolist()[::-1],
        fill="toself",
        fillcolor="rgba(82,183,136,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Normalidad (Fourier ±1.5σ)",
    ))

    # Línea Fourier base
    fig.add_trace(go.Scatter(
        x=meses_labels, y=profile["fourier"],
        mode="lines", name="Base Fourier",
        line=dict(color="#f4a261", width=2, dash="dash"),
    ))

    # Línea de consumo histórico real
    fig.add_trace(go.Scatter(
        x=meses_labels, y=profile["consumo_ratio_medio"],
        mode="lines+markers", name="Histórico Real",
        line=dict(color="#e0e0e0", width=2),
        marker=dict(size=5, color="#e0e0e0"),
    ))

    # Punto simulado en el mes elegido
    if mes_simulacion is not None:
        idx_mes = mes_simulacion - 1
        if 0 <= idx_mes < len(meses_labels):
            nivel, color_n, emoji_n = _get_alert_info(
                (consumo_sim - engine.fourier_monthly.get(mes_simulacion, engine.fourier_base)) / engine.sigma_residuo
                if engine.sigma_residuo > 0 else 0.0
            )
            fig.add_trace(go.Scatter(
                x=[meses_labels[idx_mes]],
                y=[consumo_sim],
                mode="markers+text",
                name="Escenario Simulado",
                marker=dict(size=18, color=color_n, symbol="star",
                            line=dict(width=2, color="#fff")),
                text=[f"{emoji_n} {consumo_sim:.2f}"],
                textposition="top center",
                textfont=dict(color=color_n, size=11),
                hovertemplate=f"<b>{MESES_ES.get(mes_simulacion, '')}</b><br>Consumo sim: {consumo_sim:.2f} m³/cto<br>{nivel}<extra></extra>",
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="m³ / contrato", gridcolor="rgba(255,255,255,0.07)", rangemode="tozero"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def _build_waterfall_chart(engine: WhatIfEngine, result: dict) -> go.Figure:
    """
    Waterfall chart que muestra cómo se construye el consumo simulado:
    Base Fourier → (+/-) cada feature → Consumo Simulado.
    """
    if not result["delta_por_feature"]:
        return go.Figure()

    labels  = ["Base Fourier"]
    values  = [result["fourier_mes"]]
    measure = ["absolute"]
    colors  = ["#4cc9f0"]

    for f in FEATURES_WHATIF:
        col = f["col"]
        delta = result["delta_por_feature"].get(col, None)
        if delta is None:
            continue
        labels.append(f["icon"] + " " + f["label"].split("(")[0].strip())
        values.append(delta)
        measure.append("relative")
        colors.append(f["color"] if delta >= 0 else "#636363")

    labels.append("Consumo Simulado")
    values.append(result["consumo_sim"])
    measure.append("total")
    colors.append(_get_alert_info(result["z_sim"])[1])

    fig = go.Figure(go.Waterfall(
        name="Consumo",
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        text=[f"{v:+.2f}" if m == "relative" else f"{v:.2f} m³" for v, m in zip(values, measure)],
        textposition="outside",
        connector={"line": {"color": "rgba(255,255,255,0.15)"}},
        increasing={"marker": {"color": "#e74c3c"}},
        decreasing={"marker": {"color": "#52b788"}},
        totals={"marker": {"color": _get_alert_info(result["z_sim"])[1]}},
        hoverinfo="x+y",
    ))
    # Colorear la barra de Base con azul
    fig.data[0].textfont = dict(color="#e0e0e0", size=10)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="m³ / contrato", gridcolor="rgba(255,255,255,0.07)"),
        xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.03)"),
        height=290,
        showlegend=False,
    )
    return fig


def _render_plausibilidad(plaus: dict) -> None:
    """Muestra el indicador de plausibilidad del escenario."""
    nivel = plaus.get("nivel")
    dist  = plaus.get("distancia")
    pct   = plaus.get("percentil")

    if nivel == "sin_datos" or nivel is None:
        return

    config = {
        "plausible":  ("🟢", "#52b788", "Escenario Plausible",
                       "La combinación de features es consistente con el histórico del barrio."),
        "poco_comun": ("🟡", "#f39c12", "Escenario Poco Común",
                       f"Este escenario ocurre en menos del {100 - (pct or 90)}% de los períodos históricos."),
        "atipico":    ("🟠", "#e67e22", "Escenario Atípico",
                       f"La combinación de variables es inusual (top {100 - (pct or 95)}% de rareza histórica)."),
        "extremo":    ("🔴", "#e74c3c", "Escenario Extremo",
                       "Esta combinación de features nunca se ha observado en el período analizado."),
    }
    emoji, color, titulo, desc = config.get(nivel, ("⚪", "#888", "Sin datos", ""))

    dist_str = f" · Distancia Mahalanobis: {dist:.2f}" if dist is not None else ""
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.03); border-left:3px solid {color};
        padding:8px 14px; border-radius:5px; margin-top:6px; font-size:13px;">
            <span style="color:{color}; font-weight:700;">{emoji} {titulo}</span>
            <span style="color:#aaa; margin-left:6px;">{desc}{dist_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# RENDER PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def render_whatif(df: pd.DataFrame, barrio: str | None = None) -> None:
    """
    Renderiza el simulador What-If v2 completo en el tab de Streamlit.

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
            Ajusta los sliders para simular un consumo hipotético. El sistema utiliza
            <b>betas no-lineales por tramo</b> para capturar mejor la relación entre
            factores externos y consumo, y escala el resultado según la <b>estacionalidad Fourier</b>
            del mes seleccionado.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("No hay datos disponibles para el filtro temporal y de barrio seleccionado.")
        return

    # ── Motor (cacheado por barrio) ───────────────────────────────────────────
    engine = _get_engine(df, barrio)
    
    # Extraer pre-calculo del mes directamente de session_state para iniciar los sliders al mes correcto
    mes_str = st.session_state.get(f"whatif_mes_{titulo_barrio}", "📅 Media anual")
    mes_activo = None
    if mes_str != "📅 Media anual":
        try:
            mes_activo = int(mes_str.split("(")[1].rstrip(")"))
        except Exception:
            pass
            
    feat_means = engine.get_feat_means(mes_activo)

    # ── Layout: sliders izquierda | resultados derecha ────────────────────────
    col_sliders, col_results = st.columns([1, 1.4], gap="large")

    with col_sliders:
        st.markdown("#### ⚙️ Parámetros del Escenario")

        # Selector de mes de simulación
        meses_opciones = ["📅 Media anual"] + [f"{MESES_ES[m]} ({m})" for m in range(1, 13)]
        mes_sel_label = st.select_slider(
            "Mes de simulación",
            options=meses_opciones,
            value="📅 Media anual",
            key=f"whatif_mes_{titulo_barrio}",
            help="Ancla la simulación a un mes concreto del año para tener en cuenta la estacionalidad.",
        )
        if mes_sel_label == "📅 Media anual":
            mes_simulacion = None
        else:
            # Extraer número de mes del label "Enero (1)"
            try:
                mes_simulacion = int(mes_sel_label.split("(")[1].rstrip(")"))
            except Exception:
                mes_simulacion = None

        st.markdown("---")

        feature_values: dict[str, float] = {}
        for f in FEATURES_WHATIF:
            col     = f["col"]
            mu_x    = feat_means.get(col, (f["min"] + f["max"]) / 2)
            default = round(mu_x / f["step"]) * f["step"]
            default = float(np.clip(default, f["min"], f["max"]))

            # Añadimos el mes al key para que los sliders adopten el valor por defecto del nuevo mes
            key_din = f"whatif_{col}_{titulo_barrio}_{mes_sel_label}"

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
                key=key_din,
                label_visibility="collapsed",
            )
            feature_values[col] = st.session_state.get(key_din, val)

        st.markdown("---")
        c1, c2 = st.columns(2)
        if c1.button("↩ Restaurar", key="whatif_reset", type="secondary", use_container_width=True):
            # Limpiar todos los parámetros del simulador para este barrio (incluyendo cualquier mes cacheado)
            keys_to_del = [k for k in st.session_state.keys() if k.startswith("whatif_") and titulo_barrio in k]
            for k in keys_to_del:
                del st.session_state[k]
            st.rerun()

        if c2.button("📋 Ver betas", key="whatif_betas_toggle", type="secondary", use_container_width=True):
            st.session_state["whatif_show_betas"] = not st.session_state.get("whatif_show_betas", False)

        if st.session_state.get("whatif_show_betas", False):
            rows_b = []
            for f in FEATURES_WHATIF:
                col = f["col"]
                if col not in engine.betas:
                    continue
                rows_b.append({
                    "Feature": f["icon"] + " " + f["label"].split("(")[0].strip(),
                    "β_bajo":  f"{engine.betas[col].get('bajo', 0):.4f}",
                    "β_medio": f"{engine.betas[col].get('medio', 0):.4f}",
                    "β_alto":  f"{engine.betas[col].get('alto', 0):.4f}",
                })
            if rows_b:
                st.caption("Betas no-lineales del barrio (bajo/medio/alto percentil):")
                st.dataframe(pd.DataFrame(rows_b), hide_index=True, use_container_width=True)

    # ── Simulación ─────────────────────────────────────────────────────────────
    result    = engine.simulate(feature_values, mes_simulacion)
    consumo_sim      = result["consumo_sim"]
    z_sim            = result["z_sim"]
    fourier_mes      = result["fourier_mes"]
    factor_estacional= result["factor_estacional"]
    nivel, color_nivel, emoji_nivel = _get_alert_info(z_sim)

    # ── Panel de Resultados ───────────────────────────────────────────────────
    with col_results:
        st.markdown("#### 📊 Resultado de la Simulación")

        # KPIs rápidos
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Consumo Simulado",
            f"{consumo_sim:.2f} m³/cto",
            delta=f"{consumo_sim - fourier_mes:+.2f} vs base",
            delta_color="inverse",
        )
        k2.metric("Z-Score", f"{z_sim:.3f} σ", delta=nivel, delta_color="off")
        k3.metric(
            "Factor Estac.",
            f"×{factor_estacional:.2f}",
            delta=f"{MESES_ES.get(mes_simulacion, 'Anual')}",
            delta_color="off",
        )

        # Plausibilidad
        _render_plausibilidad(result["plausibilidad"])

        # Gauge
        fig_gauge = _build_gauge(z_sim, nivel, color_nivel)
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_whatif")

        # Banner de alerta
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
                        — El escenario supera el umbral estadístico de anomalía
                        (|z| &gt; {Z_LEVE if abs(z_sim) <= Z_MOD else Z_MOD if abs(z_sim) <= Z_GRAVE else Z_GRAVE}σ).
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

    # ── Fila inferior: Waterfall (ancho completo) ─────────────────────────────
    st.markdown("---")
    st.markdown("##### Construcción del Consumo Simulado")
    st.caption("Cada feature suma o resta al valor base Fourier.")
    fig_wfall = _build_waterfall_chart(engine, result)
    st.plotly_chart(fig_wfall, use_container_width=True, key="wfall_whatif")

    # ── Perfil anual ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Perfil Anual del Barrio con Punto Simulado")
    st.caption(
        f"Línea blanca = consumo histórico real (media mensual). "
        f"{'Estrella = escenario simulado en ' + MESES_ES.get(mes_simulacion, '') if mes_simulacion else 'Selecciona un mes para anclar el escenario.'}"
    )
    fig_annual = _build_annual_profile(engine, consumo_sim, mes_simulacion)
    st.plotly_chart(fig_annual, use_container_width=True, key="annual_whatif")

    # ── Tabla de features ──────────────────────────────────────────────────────
    with st.expander("📋 Ver valores del escenario simulado vs. histórico"):
        rows = []
        for f in FEATURES_WHATIF:
            col   = f["col"]
            val_s = feature_values.get(col, 0.0)
            val_h = feat_means.get(col, 0.0)
            std_h = engine.feat_stats.get(col, {}).get("std", 1.0)
            delta_sigma = (val_s - val_h) / std_h if std_h > 0 else 0.0
            delta_pct   = ((val_s - val_h) / val_h * 100) if val_h != 0 else 0.0
            tramo = "bajo" if val_s <= engine.feat_stats.get(col, {}).get("q33", val_h) else (
                    "medio" if val_s <= engine.feat_stats.get(col, {}).get("q66", val_h) else "alto")
            beta  = engine.betas.get(col, {}).get(tramo, 0.0)
            rows.append({
                "Feature":            f["icon"] + " " + f["label"],
                "Histórico (media)":  f"{val_h:.3f} {f['unit']}",
                "Simulado":           f"{val_s:.3f} {f['unit']}",
                "Δ (%)":              f"{delta_pct:+.1f}%",
                "Δ (σ)":              f"{delta_sigma:+.2f}σ",
                "Tramo β":            tramo,
                "β activo":           f"{beta:.4f}",
                "Δ consumo":          f"{result['delta_por_feature'].get(col, 0.0):+.3f} m³/cto",
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

    # ── Nota metodológica ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="color:#555; font-size:11px; margin-top:18px; line-height:1.6;">
            <b>Nota metodológica v2:</b> El simulador utiliza <b>betas no-lineales por tramo</b>:
            la pendiente de impacto se calcula por separado en el tercio bajo, medio y alto de la
            distribución histórica de cada feature, capturando relaciones tipo umbral (p. ej., temperatura
            muy alta → efecto desproporcionado). El delta se <b>escala por la estacionalidad Fourier</b>
            del mes seleccionado (mismo Δ temperatura impacta más en verano que en invierno).
            El <b>Score de Plausibilidad</b> usa la distancia de Mahalanobis al centroide histórico
            para advertir si el escenario es inusual o nunca observado en el barrio.
            Umbrales: |z| &gt; 1.5 → Leve · |z| &gt; 2.0 → Moderado · |z| &gt; 2.5 → Grave.
        </div>
        """,
        unsafe_allow_html=True,
    )
