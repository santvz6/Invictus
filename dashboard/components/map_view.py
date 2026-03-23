"""
map_view.py
-----------
Componente del mapa principal: Folium Choropleth + HeatMap plugin.
Retorna el barrio seleccionado al hacer clic en un polígono.
"""

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

from src.config import DatasetKeys

ALICANTE_CENTER = [38.3452, -0.4810]
ALICANTE_ZOOM   = 13

# Paleta de colores por intensidad (fría → caliente)
COLORMAP_STEPS  = ["#0d1b2a", "#1b4965", "#2d6a4f", "#52b788", "#d9ed92",
                   "#f4a261", "#e76f51", "#c1121f"]


def render_map(df_barrio: pd.DataFrame, feature_col: str, gdf=None) -> dict:
    """
    Dibuja el mapa de calor interactivo.

    Parameters
    ----------
    df_barrio : pd.DataFrame  — 1 fila por barrio (ya agregado)
    feature_col : str         — Columna a usar como intensidad del calor
    gdf : GeoDataFrame | None — Geometrías reales de barrios (puede ser None)

    Returns
    -------
    dict  — Salida de st_folium (contiene last_clicked, last_active_drawing, etc.)
    """
    m = folium.Map(
        location=ALICANTE_CENTER,
        zoom_start=ALICANTE_ZOOM,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # ── Capa Choropleth (si tenemos GeoDataFrame real) ─────────────────
    choropleth_success = False
    if gdf is not None and not gdf.empty and feature_col in df_barrio.columns:
        try:
            choropleth_success = _add_choropleth(m, gdf, df_barrio, feature_col)
        except Exception:
            choropleth_success = False
    
    if not choropleth_success:
        # ── Fallback: HeatMap con coordenadas aproximadas por barrio ──
        _add_heatmap_fallback(m, df_barrio, feature_col)

    # ── Leyenda flotante ───────────────────────────────────────────────
    _add_legend(m, feature_col, df_barrio.get(feature_col, pd.Series([0])))

    return st_folium(
        m, 
        use_container_width=True, 
        height=650, 
        returned_objects=["last_active_drawing", "last_clicked"]
    )


def _add_choropleth(m, gdf, df_barrio, feature_col) -> bool:
    """Añade capa Choropleth usando geometrías reales + datos. Retorna True si tuvo éxito."""
    import branca.colormap as cm

    # Merge geometría + datos
    gdf_merged = gdf.copy()
    
    # 1. Estandarización robusta y filtro antaisolape masivo
    if "barrio_id" in gdf_merged.columns:
        gdf_merged["barrio_limpio"] = gdf_merged["barrio_id"]
    else:
        gdf_merged["barrio_limpio"] = gdf_merged["DENOMINACI"].str.strip().str.upper() \
            if "DENOMINACI" in gdf_merged.columns else gdf_merged.iloc[:, 0].astype(str).str.strip().str.upper()
            
    gdf_merged = gdf_merged[~gdf_merged["barrio_limpio"].isin(["ALICANTE", "ALACANT", "ALICANTE/ALACANT"])]

    from src.config import DatasetKeys
    df_temp = df_barrio.copy()
    df_temp["barrio_limpio"] = df_temp[DatasetKeys.BARRIO].str.split("-", n=1).str[-1].str.strip().str.upper()

    # FIX: Un inner merge asegura que eliminemos los polígonos que NO están en nuestros datos 
    # protegiendo los barrios que realmente utilizamos.
    gdf_merged = gdf_merged.merge(df_temp, on="barrio_limpio", how="inner")
    
    if gdf_merged.empty:
        return False
        
    gdf_merged[feature_col] = gdf_merged[feature_col].fillna(0)
    
    for col in [DatasetKeys.RECONSTRUCTION_ERROR, DatasetKeys.ALERTA_TURISTICA_ILEGAL]:
        if col not in gdf_merged.columns:
            gdf_merged[col] = 0.0
            
    if DatasetKeys.RECONSTRUCTION_ERROR in gdf_merged.columns:
        gdf_merged[DatasetKeys.RECONSTRUCTION_ERROR] = gdf_merged[DatasetKeys.RECONSTRUCTION_ERROR].round(3)

    # 2. ESCALADO ROBUSTO CONTRA OUTLIERS
    vmin = gdf_merged[feature_col].quantile(0.05)
    vmax = gdf_merged[feature_col].quantile(0.95)
    if vmin == vmax:
        vmin, vmax = vmin * 0.9, vmax * 1.1 + 1e-5

    colormap = cm.LinearColormap(
        colors=["#0d1b2a", "#1b4965", "#4cc9f0", "#f39c12", "#e74c3c"],
        vmin=vmin, vmax=vmax,
        caption=feature_col,
    )

    def style_fn(feature):
        val = feature["properties"].get(feature_col, 0) or 0
        return {
            "fillColor":   colormap(val),
            "color":       "rgba(255, 255, 255, 0.4)",
            "weight":      1,
            "fillOpacity": 0.75,
        }

    def highlight_fn(_):
        return {"weight": 3, "color": "#4cc9f0", "fillOpacity": 0.95}

    # 3. FIX HOVER Y SOLAPE: Se unifica el Tooltip y el Style en una sola capa
    folium.GeoJson(
        gdf_merged.__geo_interface__,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["barrio_limpio", feature_col,
                    DatasetKeys.RECONSTRUCTION_ERROR, DatasetKeys.ALERTA_TURISTICA_ILEGAL],
            aliases=["Barrio:", f"{feature_col}:", "Error reconstrucción:", "Alertas:"],
            localize=True,
            style=("background-color: #0d1b2a; color: #e0e0e0; font-family: 'Inter', sans-serif; "
                   "font-size: 13px; padding: 10px; border: 1px solid #4cc9f0; border-radius: 6px;")
        ),
        popup=folium.GeoJsonPopup(fields=["barrio_limpio"], aliases=["Barrio:"]),
    ).add_to(m)

    colormap.add_to(m)
    return True


def _add_heatmap_fallback(m, df_barrio, feature_col):
    """
    Fallback: posiciona los barrios con coordenadas aproximadas centradas en Alicante
    y dibuja un HeatMap interpolado.
    """
    # Coordenadas aproximadas de barrios conocidos (lat, lon)
    BARRIO_COORDS = {
        "PLAYA SAN JUAN":         (38.3768, -0.4326),
        "VISTAHERMOSA":           (38.3700, -0.4400),
        "ZONA TURISTICA":         (38.3560, -0.4800),
        "FLORIDA BAJA":           (38.3350, -0.4900),
        "FLORIDA ALTA":           (38.3280, -0.4860),
        "BENALUA":                (38.3410, -0.4860),
        "CAROLINAS ALTAS":        (38.3490, -0.4750),
        "CAROLINAS BAJAS":        (38.3460, -0.4780),
        "CASCO ANTIGUO":          (38.3450, -0.4810),
        "GRAN VIA":               (38.3430, -0.4840),
        "MERCADO":                (38.3440, -0.4830),
        "SAN BLAS":               (38.3380, -0.4960),
        "CIUDAD JARDIN":          (38.3520, -0.4990),
        "LOS ANGELES":            (38.3300, -0.4920),
        "RABASA":                 (38.3260, -0.4840),
        "MONTE TOSSAL":           (38.3480, -0.4850),
        "EL PALMERAL":            (38.3600, -0.4700),
        "TÓMBOLA":                (38.3650, -0.4620),
        "VIRGEN DEL REMEDIO":     (38.3320, -0.5010),
        "JUAN XXIII":             (38.3290, -0.5050),
    }

    if feature_col not in df_barrio.columns:
        return

    series = df_barrio[feature_col].replace([np.inf, -np.inf], np.nan).fillna(0)
    vmax = series.max()
    if pd.isna(vmax) or vmax == 0:
        vmax = 1

    heat_data = []
    for _, row in df_barrio.iterrows():
        barrio = row.get("barrio", row.iloc[0])
        lat, lon = BARRIO_COORDS.get(barrio, (
            ALICANTE_CENTER[0] + np.random.uniform(-0.04, 0.04),
            ALICANTE_CENTER[1] + np.random.uniform(-0.04, 0.04),
        ))
        weight = float(row[feature_col]) / vmax if vmax else 0
        heat_data.append([lat, lon, weight])

    if heat_data:
        HeatMap(
            heat_data,
            radius=35, blur=25,
            gradient={0.2: "#1b4965", 0.5: "#52b788", 0.75: "#f4a261", 1.0: "#c1121f"},
        ).add_to(m)

    # Marcadores para poder clicar
    for _, row in df_barrio.iterrows():
        barrio = row.get("barrio", row.iloc[0])
        lat, lon = BARRIO_COORDS.get(barrio, (
            ALICANTE_CENTER[0] + np.random.uniform(-0.04, 0.04),
            ALICANTE_CENTER[1] + np.random.uniform(-0.04, 0.04),
        ))
        val = row.get(feature_col, 0)
        anomalias = int(row.get(DatasetKeys.ALERTA_TURISTICA_ILEGAL, 0))
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="#c1121f" if anomalias > 0 else "#52b788",
            fill=True, fill_opacity=0.8,
            tooltip=f"<b>{barrio}</b><br>{feature_col}: {val:.1f}<br>Alertas: {anomalias}",
            popup=barrio,
        ).add_to(m)


def _add_legend(m, feature_col: str, series: pd.Series):
    """Añade una leyenda HTML flotante en esquina superior derecha."""
    lo = float(series.min()) if len(series) > 0 else 0
    hi = float(series.max()) if len(series) > 0 else 1
    html = f"""
    <div style="
        position: fixed; top: 10px; right: 10px; z-index: 1000;
        background: rgba(13,27,42,0.85); color: #e0e0e0;
        padding: 10px 16px; border-radius: 10px;
        font-family: 'Inter', sans-serif; font-size: 12px;
        backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.12);
    ">
        <b style="font-size:13px;">🌡 {feature_col}</b><br>
        <div style="display:flex; align-items:center; gap:6px; margin-top:6px;">
            <span style="font-size:10px;">{lo:.1f}</span>
            <div style="height:10px; width:90px; background:linear-gradient(to right,#1b4965,#52b788,#f4a261,#c1121f);
                        border-radius:3px;"></div>
            <span style="font-size:10px;">{hi:.1f}</span>
        </div>
        <div style="margin-top:6px; font-size:10px; color:#c1121f;">● Alerta turística</div>
        <div style="font-size:10px; color:#52b788;">● Sin alerta</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))
