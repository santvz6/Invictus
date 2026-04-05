"""
data_loader.py
--------------
Carga los datos procesados del pipeline Water2Fraud.
Si el CSV procesado no existe todavía, genera datos sintéticos de demostración
para que el dashboard sea funcional en todo momento.
"""

import sys
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import Paths, DatasetKeys, get_logger

Paths.init_project()
logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Lista oficial de barrios de Alicante (AMAEM)
# ──────────────────────────────────────────────
BARRIOS_ALICANTE = [
    "FLORIDA BAJA", "FLORIDA ALTA", "CIUDAD DE ASIS", "JUAN XXIII",
    "BENALUA", "CAROLINAS ALTAS", "CAROLINAS BAJAS", "CIUDAD JARDIN",
    "CASTILLO-SANTA BARBARA", "COLONIA REQUENA", "CUATROCIENTAS VIVIENDAS",
    "EL PALMERAL", "EL PINO", "GRAN VIA", "LOS ANGELES", "MERCADO",
    "MONTE TOSSAL", "NUEVA DENIA", "PINOS GENIL", "PLAYA SAN JUAN",
    "RABASA", "SAGRADA FAMILIA", "SAN ANTOLIN", "SAN BLAS",
    "TÓMBOLA", "VISTAHERMOSA", "VIRGEN DEL REMEDIO", "ZONA TURISTICA",
    "EL RAVAL", "CASCO ANTIGUO",
]

FEATURES_DISPONIBLES = {
    "Consumo Total (m³)":          DatasetKeys.CONSUMO,
    "Nº Contratos":                DatasetKeys.NUM_CONTRATOS,
    "Ratio Consumo/Contrato":      DatasetKeys.CONSUMO_RATIO,
    "Pernoctaciones Turísticas":   DatasetKeys.PERNOCT_VT_PROV_INE,
    "Z-Score Residual Físico":     DatasetKeys.Z_ERROR_FINAL,
}


# ──────────────────────────────────────────────
# Carga / Generación de datos
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    """
    Intenta cargar el CSV procesado. Si no existe, genera datos mock.
    Devuelve un DataFrame con todos los meses de 2022-2024 y todos los barrios.
    """
    csv_path = Paths.PROC_CSV_AMAEM_NOT_SCALED

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if DatasetKeys.FECHA in df.columns:
            df[DatasetKeys.FECHA] = pd.to_datetime(df[DatasetKeys.FECHA], errors="coerce")
        return df
    else:
        logger.warning("No se encontró el CSV procesado. Generando datos sintéticos de demostración.")
        sys.exit(1)


@st.cache_data(show_spinner=False)
def load_geodataframe() -> gpd.GeoDataFrame | None:
    """
    Carga el GeoJSON de entidades de población (barrios con geometría real).
    Devuelve None si el archivo no existe.
    """
    geojson_path = Paths.RAW_JSON_ENTIDADES_POBLACION
    if not geojson_path.exists():
        return None
    try:
        gdf = gpd.read_file(geojson_path)
        gdf = gdf.to_crs(epsg=4326)
        
        # Estandarizamos el nombre del barrio en el GeoJSON (Vital para tooltips y merge correctos sin solapes)
        for col in ["barrio_limpio", "DENOMINACI", "barrio"]:
            if col in gdf.columns:
                gdf["barrio_id"] = gdf[col].astype(str).str.strip().str.upper()
                break
        else:
            # Fallback si no existe ninguna de las anteriores
            gdf["barrio_id"] = "DESCONOCIDO"
            
        # Eliminamos el polígono que representa a toda la ciudad para que no cubra a los demás barrios
        gdf = gdf[~gdf["barrio_id"].isin(["ALICANTE", "ALACANT", "ALICANTE/ALACANT"])]
            
        return gdf
    except Exception:
        return None


def filter_dataframe(df: pd.DataFrame, fecha_inicio, fecha_fin, 
                     barrio_filter: str | None = None,
                     uso_filter: str | None = None) -> pd.DataFrame:
    """Filtra el DataFrame por rango de fechas, barrio y uso."""
    mask = (df[DatasetKeys.FECHA] >= pd.Timestamp(fecha_inicio)) & \
           (df[DatasetKeys.FECHA] <= pd.Timestamp(fecha_fin))
    df_filtered = df[mask]
    
    if barrio_filter and barrio_filter != "Todos los barrios":
        df_filtered = df_filtered[df_filtered[DatasetKeys.BARRIO] == barrio_filter]
        
    if uso_filter and uso_filter != "Todos los usos":
        df_filtered = df_filtered[df_filtered[DatasetKeys.USO] == uso_filter]
        
    return df_filtered


def aggregate_by_barrio(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el DataFrame filtrado a nivel de barrio (una fila por barrio)."""
    
    df_copy = df.copy()
    if DatasetKeys.ALERTA_NIVEL in df_copy.columns:
        df_copy["num_alertas"] = (df_copy[DatasetKeys.ALERTA_NIVEL] != "Normal").astype(int)
        
    agg = {
        DatasetKeys.CONSUMO:                  "sum",
        DatasetKeys.NUM_CONTRATOS:            "mean",
        DatasetKeys.CONSUMO_RATIO:            "mean",
        DatasetKeys.OCUP_VT_PROV_INE:        "mean",
        DatasetKeys.PERNOCT_VT_PROV_INE:     "mean",
        DatasetKeys.DIAS_FESTIVOS:            "mean",
        DatasetKeys.TEMP_MEDIA:               "mean",
        DatasetKeys.PRECIPITACION:            "mean",
        DatasetKeys.CONSUMO_FISICO_ESPERADO:  "sum",
        DatasetKeys.PREDICCION_FOURIER:       "sum",
        DatasetKeys.Z_ERROR_FINAL:            "max",   # Mostrar max Z-score en lugar de mean que se cancela a 0
        "num_alertas":                        "sum",
    }
    # Filtramos columnas que existan
    agg = {k: v for k, v in agg.items() if k in df_copy.columns}
    
    df_agg = df_copy.groupby(DatasetKeys.BARRIO).agg(agg).reset_index()
    
    # Limpiamos Infs y NaNs que rompen el mapa de Folium
    df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_agg.fillna(0, inplace=True)
    
    # Redondeo seguro para que los Tooltips no muestren números con decimales infinitos
    num_cols = df_agg.select_dtypes(include=[np.number]).columns
    df_agg[num_cols] = df_agg[num_cols].round(2)

    # ── FIX: Convertir todos los tipos numpy a tipos Python nativos.
    # numpy.int64 / float64 no son serializables por el JSON nativo de Python 3.14
    # que usa Jinja2/branca. Esto afecta al colormap, los tooltips y el GeoJson.
    for col in df_agg.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        df_agg[col] = df_agg[col].astype(int)
    for col in df_agg.select_dtypes(include=["float64", "float32"]).columns:
        df_agg[col] = df_agg[col].astype(float)

    return df_agg
