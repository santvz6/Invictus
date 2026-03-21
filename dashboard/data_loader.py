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
from src.config import Paths, DatasetKeys

Paths.init_project()

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
    "Consumo Total (m³)":        DatasetKeys.CONSUMO,
    "Nº Contratos":              DatasetKeys.NUM_CONTRATOS,
    "Ratio Consumo/Contrato":    DatasetKeys.CONSUMO_RATIO,
    "% Viviendas Turísticas":    DatasetKeys.PCT_VT_BARRIO,
    "Nº VT por Barrio":          DatasetKeys.NUM_VT_BARRIO,
    "Temperatura Media (°C)":    DatasetKeys.TEMP_MEDIA,
    "Precipitación (mm)":        DatasetKeys.PRECIPITACION,
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
        # Asegurar columna fecha como datetime
        if DatasetKeys.FECHA in df.columns:
            df[DatasetKeys.FECHA] = pd.to_datetime(df[DatasetKeys.FECHA], errors="coerce")
        return df

    # ── Datos sintéticos ──────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)

    fechas = pd.date_range("2022-01-01", "2024-12-01", freq="MS")
    rows = []

    for barrio in BARRIOS_ALICANTE:
        # Perfil base por barrio (turístico vs residencial)
        es_turistico = barrio in {"PLAYA SAN JUAN", "VISTAHERMOSA", "ZONA TURISTICA",
                                   "CASTILLO-SANTA BARBARA", "CASCO ANTIGUO"}
        base_consumo  = np.random.uniform(8_000, 25_000) if es_turistico else np.random.uniform(2_000, 12_000)
        base_contratos = int(np.random.uniform(80, 250) if es_turistico else np.random.uniform(20, 120))
        pct_vt = np.random.uniform(15, 45) if es_turistico else np.random.uniform(1, 14)

        for fecha in fechas:
            mes = fecha.month
            # Estacionalidad sinusoidal (pico en verano)
            estacional = 1 + 0.5 * np.sin((mes - 3) * np.pi / 6)
            consumo = base_consumo * estacional * np.random.normal(1, 0.08)
            temp_media = 18 + 8 * np.sin((mes - 2) * np.pi / 6) + np.random.normal(0, 1.5)
            precip = max(0, 30 - 25 * np.sin((mes - 3) * np.pi / 6) + np.random.normal(0, 8))
            consumo_fisico = consumo * (1 + 0.03 * (temp_media - 18)) * np.random.normal(1, 0.05)

            # Anomalías artificiales (10% de los registros turísticos)
            is_anomaly = es_turistico and np.random.random() < 0.10
            if is_anomaly:
                consumo *= np.random.uniform(1.6, 2.5)

            rows.append({
                DatasetKeys.BARRIO:               barrio,
                DatasetKeys.FECHA:                fecha,
                DatasetKeys.CONSUMO:              round(consumo, 1),
                DatasetKeys.NUM_CONTRATOS:        base_contratos + int(np.random.normal(0, 5)),
                DatasetKeys.CONSUMO_RATIO:        round(consumo / base_contratos, 3),
                DatasetKeys.PCT_VT_BARRIO:        round(pct_vt + np.random.normal(0, 1), 2),
                DatasetKeys.NUM_VT_BARRIO:        int(pct_vt * base_contratos / 100),
                DatasetKeys.TEMP_MEDIA:           round(temp_media, 1),
                DatasetKeys.PRECIPITACION:        round(precip, 1),
                DatasetKeys.CONSUMO_FISICO_ESPERADO: round(consumo_fisico, 1),
                DatasetKeys.PREDICCION_FOURIER:   round(consumo_fisico * 0.95, 1),
                "reconstruction_error":           round(abs(consumo - consumo_fisico) / consumo_fisico, 4),
                "is_ae_anomaly":                  is_anomaly,
                "ALERTA_TURISTICA_ILEGAL":        is_anomaly,
                "cluster":                        int(np.random.randint(0, 3)),
            })

    df = pd.DataFrame(rows)
    df[DatasetKeys.FECHA] = pd.to_datetime(df[DatasetKeys.FECHA])
    return df


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
        return gdf
    except Exception:
        return None


def filter_dataframe(df: pd.DataFrame, fecha_inicio, fecha_fin, contrato_filter: str | None = None) -> pd.DataFrame:
    """Filtra el DataFrame por rango de fechas y opcionalmente por barrio."""
    mask = (df[DatasetKeys.FECHA] >= pd.Timestamp(fecha_inicio)) & \
           (df[DatasetKeys.FECHA] <= pd.Timestamp(fecha_fin))
    df_filtered = df[mask]
    if contrato_filter and contrato_filter != "Todos los barrios":
        df_filtered = df_filtered[df_filtered[DatasetKeys.BARRIO] == contrato_filter]
    return df_filtered


def aggregate_by_barrio(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el DataFrame filtrado a nivel de barrio (una fila por barrio)."""
    agg = {
        DatasetKeys.CONSUMO:                  "sum",
        DatasetKeys.NUM_CONTRATOS:            "sum",
        DatasetKeys.CONSUMO_RATIO:            "mean",
        DatasetKeys.PCT_VT_BARRIO:            "mean",
        DatasetKeys.NUM_VT_BARRIO:            "sum",
        DatasetKeys.TEMP_MEDIA:               "mean",
        DatasetKeys.PRECIPITACION:            "mean",
        DatasetKeys.CONSUMO_FISICO_ESPERADO:  "sum",
        "reconstruction_error":              "mean",
        "ALERTA_TURISTICA_ILEGAL":           "sum",
    }
    # Filtramos columnas que existan
    agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.groupby(DatasetKeys.BARRIO).agg(agg).reset_index()
