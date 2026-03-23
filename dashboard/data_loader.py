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
    "% Viviendas Turísticas":    DatasetKeys.PCT_VT_BARRIO_INE,
    "Nº VT por Barrio":          DatasetKeys.NUM_VT_BARRIO_INE,
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
            
        # 1. Intentar enriquecer con los resultados del último experimento (Pipeline ML)
        if Paths.EXPERIMENTS_DIR.exists():
            experimentos = sorted([d for d in Paths.EXPERIMENTS_DIR.iterdir() if d.is_dir()])
            if experimentos:
                latest_exp = experimentos[-1]
                res_path = latest_exp / "resultados_completos_tecnicos.csv"
                if res_path.exists():
                    df_res = pd.read_csv(res_path)
                    # Quitar espacios en blanco de los nombres de columnas (Bug de Pandas espacial al guardar)
                    df_res.columns = df_res.columns.str.strip()
                    # Quitar espacios en blanco de los valores string para que el merge por 'barrio' funcione
                    # NOTA: Usamos .map en lugar de .applymap para evitar el FutureWarning de Pandas
                    df_res = df_res.map(lambda x: x.strip() if isinstance(x, str) else x)
                    
                    if DatasetKeys.FECHA in df_res.columns:
                        df_res[DatasetKeys.FECHA] = pd.to_datetime(df_res[DatasetKeys.FECHA], errors="coerce")
                        
                    cols_merge = [DatasetKeys.BARRIO, DatasetKeys.FECHA]
                    if DatasetKeys.USO in df.columns and DatasetKeys.USO in df_res.columns:
                        cols_merge.append(DatasetKeys.USO)
                        
                    cols_extract = [
                        DatasetKeys.RECONSTRUCTION_ERROR, DatasetKeys.AE_SCORE, 
                        DatasetKeys.IS_AE_ANOMALY, DatasetKeys.ALERTA_TURISTICA_ILEGAL, 
                        DatasetKeys.CLUSTER, DatasetKeys.CONSUMO_FISICO_ESPERADO, 
                        DatasetKeys.PREDICCION_FOURIER
                    ]
                    cols_extract = [c for c in cols_extract if c in df_res.columns]

                    # IMPORTANTE: Asegurar que las columnas de interés son numéricas 
                    # (el strip las puede dejar como object si venían con espacios)
                    for col in cols_extract:
                        if col in [DatasetKeys.IS_AE_ANOMALY, DatasetKeys.ALERTA_TURISTICA_ILEGAL]:
                            df_res[col] = df_res[col].map(lambda x: str(x).lower() == 'true')
                        else:
                            df_res[col] = pd.to_numeric(df_res[col], errors='coerce').fillna(0.0)
                    
                    if cols_extract:
                        df_res_sub = df_res[cols_merge + cols_extract].drop_duplicates(subset=cols_merge)
                        df = df.merge(df_res_sub, on=cols_merge, how="left")

        # 2. Parche de seguridad para el Dashboard: asegurar las columnas de inferencia
        if DatasetKeys.AE_SCORE not in df.columns:
            if DatasetKeys.CONSUMO_FISICO_ESPERADO in df.columns and DatasetKeys.CONSUMO in df.columns:
                denominador = df[DatasetKeys.CONSUMO_FISICO_ESPERADO].fillna(1).replace(0, 1)
                df[DatasetKeys.AE_SCORE] = (abs(df[DatasetKeys.CONSUMO] - df[DatasetKeys.CONSUMO_FISICO_ESPERADO]) / denominador).round(4) * 100
            else:
                df[DatasetKeys.AE_SCORE] = 0.0
        for col in [DatasetKeys.RECONSTRUCTION_ERROR, DatasetKeys.AE_SCORE]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = 0.0

        for col, default_val in [(DatasetKeys.IS_AE_ANOMALY, False), (DatasetKeys.ALERTA_TURISTICA_ILEGAL, False), (DatasetKeys.CLUSTER, 0)]:
            if col not in df.columns:
                df[col] = default_val
            df[col] = df[col].fillna(default_val).infer_objects(copy=False)
            
        if DatasetKeys.PREDICCION_FOURIER not in df.columns:
            if DatasetKeys.CONSUMO_FISICO_ESPERADO in df.columns:
                df[DatasetKeys.PREDICCION_FOURIER] = (df[DatasetKeys.CONSUMO_FISICO_ESPERADO] * 0.95).round(1)
            else:
                df[DatasetKeys.PREDICCION_FOURIER] = 0.0
                
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
                DatasetKeys.USO:                  "DOMESTICO",
                DatasetKeys.CONSUMO:              round(consumo, 1),
                DatasetKeys.NUM_CONTRATOS:        base_contratos + int(np.random.normal(0, 5)),
                DatasetKeys.CONSUMO_RATIO:        round(consumo / base_contratos, 3),
                DatasetKeys.PCT_VT_BARRIO_INE:        round(pct_vt + np.random.normal(0, 1), 2),
                DatasetKeys.NUM_VT_BARRIO_INE:        int(pct_vt * base_contratos / 100),
                DatasetKeys.TEMP_MEDIA:           round(temp_media, 1),
                DatasetKeys.PRECIPITACION:        round(precip, 1),
                DatasetKeys.CONSUMO_FISICO_ESPERADO: round(consumo_fisico, 1),
                DatasetKeys.PREDICCION_FOURIER:   round(consumo_fisico * 0.95, 1),
                DatasetKeys.RECONSTRUCTION_ERROR: round(abs(consumo - consumo_fisico) / consumo_fisico, 4),
                DatasetKeys.IS_AE_ANOMALY:        is_anomaly,
                DatasetKeys.ALERTA_TURISTICA_ILEGAL: is_anomaly,
                DatasetKeys.CLUSTER:              int(np.random.randint(0, 3)),
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
        
        # Estandarizar nombre del barrio en el GeoJSON (Vital para tooltips y merge correctos sin solapes)
        for col in ["barrio_limpio", "DENOMINACI", "barrio"]:
            if col in gdf.columns:
                gdf["barrio_id"] = gdf[col].astype(str).str.strip().str.upper()
                break
        else:
            # Fallback si no existe ninguna de las anteriores
            gdf["barrio_id"] = "DESCONOCIDO"
            
        # Eliminar el polígono que representa a toda la ciudad para que no cubra a los demás barrios
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
    agg = {
        DatasetKeys.CONSUMO:                  "sum",
        DatasetKeys.NUM_CONTRATOS:            "sum",
        DatasetKeys.CONSUMO_RATIO:            "mean",
        DatasetKeys.PCT_VT_BARRIO_INE:        "mean",
        DatasetKeys.NUM_VT_BARRIO_INE:        "sum",
        DatasetKeys.TEMP_MEDIA:               "mean",
        DatasetKeys.PRECIPITACION:            "mean",
        DatasetKeys.CONSUMO_FISICO_ESPERADO:  "sum",
        DatasetKeys.PREDICCION_FOURIER:       "sum",
        DatasetKeys.AE_SCORE:                 "mean",
        DatasetKeys.IS_AE_ANOMALY:            "sum",
        DatasetKeys.ALERTA_TURISTICA_ILEGAL:  "sum",
    }
    # Filtramos columnas que existan
    agg = {k: v for k, v in agg.items() if k in df.columns}
    
    df_agg = df.groupby(DatasetKeys.BARRIO).agg(agg).reset_index()
    # Redondeo seguro para que los Tooltips no muestren números con decimales infinitos
    num_cols = df_agg.select_dtypes(include=[np.number]).columns
    df_agg[num_cols] = df_agg[num_cols].round(2)
    return df_agg
