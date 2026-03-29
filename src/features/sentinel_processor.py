"""
Módulo para el procesamiento de datos satelitales (NDVI) provenientes de Sentinel.

Este componente permite enriquecer el dataset de consumo hídrico con el índice 
de vegetación de diferencia normalizada (NDVI), proporcionando una capa de 
información sobre la salud de la vegetación y su posible impacto en el consumo.
"""

import pandas as pd
import logging
from src.config import DatasetKeys, Paths, get_logger

logger = get_logger(__name__)

class SentinelProcessor:
    """
    Procesador especializado en la integración de métricas de Sentinel-2.
    
    Se encarga de alinear cronológicamente los datos de NDVI mediante un anclaje 
    mensual y de resolver las discrepancias de nomenclatura en los barrios para 
    asegurar un emparejamiento preciso.
    """

    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el flujo de enriquecimiento con datos de NDVI satelital.

        Args:
            df (pd.DataFrame): Dataset principal de AMAEM.

        Returns:
            pd.DataFrame: Dataset enriquecido con la columna de NDVI:
                - NDVI_SATELITE
        """
        logger.info("Iniciando procesamiento de NDVI (Sentinel)...")
        
        # 1. Preparación del esquema temporal base
        df = SentinelProcessor._prepare_base_dataframe(df)

        # 2. Carga de la fuente de datos Sentinel
        df_ndvi = SentinelProcessor._load_ndvi_data()
        if df_ndvi is None:
            df = df.drop(columns=['fecha_cruce_mensual'])
            return df

        try:
            # 3. Normalización y limpieza del dataset de NDVI
            df_ndvi, merge_keys = SentinelProcessor._prepare_ndvi_dataset(df_ndvi)

            # 4. Integración geo-temporal de los datos
            df = SentinelProcessor._merge_ndvi_data(df, df_ndvi, merge_keys)

            # 5. Finalización y persistencia
            df = SentinelProcessor._finalize_sentinel(df)

        except Exception as e:
            logger.error(f"Error crítico durante el procesamiento de Sentinel: {e}")
            if 'fecha_cruce_mensual' in df.columns:
                df = df.drop(columns=['fecha_cruce_mensual'])
            
        return df

    @staticmethod
    def _prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Añade el periodo de cruce mensual al dataframe de AMAEM."""
        df['fecha_cruce_mensual'] = df[DatasetKeys.FECHA].dt.to_period('M')
        return df

    @staticmethod
    def _load_ndvi_data() -> pd.DataFrame | None:
        """Localiza y carga el registro histórico de NDVI desde el sistema de archivos."""
        ruta_ndvi = Paths.SENTINEL_NDVI
        if not ruta_ndvi.exists():
            logger.warning(f"Archivo NDVI no localizado en {ruta_ndvi}. Se omite este enriquecimiento.")
            return None
        return pd.read_csv(ruta_ndvi)

    @staticmethod
    def _prepare_ndvi_dataset(df_ndvi: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Estandariza el dataset de NDVI resolviendo nombres de columnas y normalizando barrios.
        """
        # Identificación de la columna temporal en origen
        col_fecha = DatasetKeys.FECHA if DatasetKeys.FECHA in df_ndvi.columns else 'fecha_mes'
        df_ndvi['fecha_cruce_mensual'] = pd.to_datetime(df_ndvi[col_fecha]).dt.to_period('M')
        
        # Resolución de la columna de barrio (soporta variantes de idioma)
        col_barrio = next((c for c in df_ndvi.columns if c.lower() in ['barrio', 'neighborhood']), None)
        
        merge_keys = ['fecha_cruce_mensual']
        
        if col_barrio:
            # Sincronizamos el nombre de la columna con el estándar del proyecto
            if col_barrio != DatasetKeys.BARRIO:
                df_ndvi = df_ndvi.rename(columns={col_barrio: DatasetKeys.BARRIO})
            
            # Normalización a mayúsculas para garantizar el emparejamiento exacto
            df_ndvi[DatasetKeys.BARRIO] = df_ndvi[DatasetKeys.BARRIO].astype(str).str.upper()
            merge_keys.append(DatasetKeys.BARRIO)
        
        # Eliminación de posibles redundancias en los datos de origen
        df_ndvi = df_ndvi.drop_duplicates(subset=merge_keys)
        
        return df_ndvi, merge_keys

    @staticmethod
    def _merge_ndvi_data(df: pd.DataFrame, df_ndvi: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
        """Realiza el cruce de datos y ajusta las cabeceras resultantes."""
        # Seleccionamos solo las columnas necesarias para el cruce
        cols_clima = merge_keys + ['ndvi_satelite']
        df = pd.merge(df, df_ndvi[cols_clima], on=merge_keys, how='left')
        
        # Renombramos para asegurar la consistencia con DatasetKeys
        df = df.rename(columns={'ndvi_satelite': DatasetKeys.NDVI_SATELITE})
        return df

    @staticmethod
    def _finalize_sentinel(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia el entorno técnico y persiste el dataset intermedio."""
        df = df.drop(columns=['fecha_cruce_mensual'])
        
        ruta_csv = Paths.PROC_CSV_STEP_SENTINEL
        logger.info(f"Registrando punto de control intermedio Sentinel en {ruta_csv}")
        
        # Filtramos solo las columnas clave para el checkpoint
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA, DatasetKeys.NDVI_SATELITE]
        cols_to_save = [c for c in cols_to_save if c in df.columns]
        
        df[cols_to_save].drop_duplicates().to_csv(ruta_csv, index=False)
        logger.info("Enriquecimiento con Sentinel finalizado.")
        return df
