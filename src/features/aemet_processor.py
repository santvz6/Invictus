"""
Módulo de procesamiento de datos climáticos de la AEMET.

Este componente se encarga de la ingesta, limpieza y enriquecimiento de los datos 
de consumo hídrico con variables meteorológicas externas (temperatura y precipitación).
"""

import pandas as pd
from src.config import DatasetKeys, Paths, get_logger
logger = get_logger(__name__)

class AEMETProcessor:
    """
    Procesador especializado en el cruce de datos climáticos históricos.
    
    Permite integrar métricas de temperatura media y precipitación acumulada 
    en el dataset principal de AMAEM, realizando un acoplamiento geo-temporal 
    basado en barrios y periodos mensuales.
    """

    @staticmethod
    def process(df_amaem: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline de enriquecimiento climático mediante un flujo modular.

        Args:
            df_amaem (pd.DataFrame): Dataset base con registros de consumo de AMAEM.

        Returns:
            pd.DataFrame: Dataset enriquecido con columnas de temperatura y precipitación:
                - TEMP_MEDIA
                - PRECIPITACION
        """
        logger.info("Iniciando enriquecimiento con datos climáticos de AEMET...")
        
        # 1. Preparación del dataset base
        df_final = AEMETProcessor._prepare_base_dataframe(df_amaem)

        # 2. Carga y limpieza de datos climáticos
        df_aemet = AEMETProcessor._load_and_clean_aemet_data()
        if df_aemet is None:
            df_final = df_final.drop(columns=['fecha_cruce_mensual'])
            return df_final

        # 3. Preparación de la referencia temporal en AEMET
        df_aemet = AEMETProcessor._add_temporal_anchor(df_aemet)

        # 4. Integración de ambos datasets
        df_final = AEMETProcessor._execute_merge(df_final, df_aemet)

        # 5. Finalización: Imputación y guardado
        df_final = AEMETProcessor._finalize_data(df_final)

        logger.info("Enriquecimiento climático finalizado.")
        return df_final

    @staticmethod
    def _prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Prepara el dataset de AMAEM creando el anclaje mensual para el cruce."""
        df_final = df.copy()
        df_final[DatasetKeys.FECHA] = pd.to_datetime(df_final[DatasetKeys.FECHA])
        df_final['fecha_cruce_mensual'] = df_final[DatasetKeys.FECHA].dt.to_period('M')
        return df_final

    @staticmethod
    def _load_and_clean_aemet_data() -> pd.DataFrame | None:
        """Carga la fuente externa, normaliza cabeceras y ajusta tipos de datos."""
        ruta_aemet = Paths.AEMET_CLIMA_BARRIOS
        try:
            df_aemet = pd.read_csv(ruta_aemet)
        except FileNotFoundError:
            logger.error(f"Archivo AEMET no localizado en {ruta_aemet}. Se omite el cruce climático.")
            return None

        # Normalización de cabeceras
        df_aemet.columns = df_aemet.columns.str.strip().str.lower().str.replace(' ', '_')

        # Mapeo según el esquema de claves
        df_aemet = df_aemet.rename(columns={
            'zona': DatasetKeys.BARRIO,
            'tm_mes': DatasetKeys.TEMP_MEDIA,
            'p_mes': DatasetKeys.PRECIPITACION
        })
            
        # Corrección de separadores decimales (coma a punto)
        for col in [DatasetKeys.TEMP_MEDIA, DatasetKeys.PRECIPITACION]:
            if col in df_aemet.columns and df_aemet[col].dtype == object:
                df_aemet[col] = df_aemet[col].str.replace(',', '.', regex=False).astype(float)
        
        return df_aemet

    @staticmethod
    def _add_temporal_anchor(df_aemet: pd.DataFrame) -> pd.DataFrame:
        """Genera el anclaje temporal en los datos de AEMET usando la clave de fecha."""
        if DatasetKeys.FECHA in df_aemet.columns:
            df_aemet[DatasetKeys.FECHA] = pd.to_datetime(df_aemet[DatasetKeys.FECHA])
            df_aemet['fecha_cruce_mensual'] = df_aemet[DatasetKeys.FECHA].dt.to_period('M')
            df_aemet = df_aemet.drop(columns=[DatasetKeys.FECHA])
        else:
            logger.warning("Referencia temporal no encontrada en el archivo de AEMET.")
        return df_aemet

    @staticmethod
    def _execute_merge(df_final: pd.DataFrame, df_aemet: pd.DataFrame) -> pd.DataFrame:
        """Realiza la integración de datos con lógica de fallback si falta el barrio."""
        if DatasetKeys.BARRIO in df_aemet.columns:
            df_aemet[DatasetKeys.BARRIO] = df_aemet[DatasetKeys.BARRIO].astype(str).str.upper()
            df_final = pd.merge(
                df_final, df_aemet,
                on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'],
                how='left'
            )
        elif 'fecha_cruce_mensual' in df_aemet.columns:
            df_final = pd.merge(df_final, df_aemet, on='fecha_cruce_mensual', how='left')
            logger.warning("fallback: No se encontró la columna de barrio en el archivo de AEMET.")
        
        return df_final

    @staticmethod
    def _finalize_data(df_final: pd.DataFrame) -> pd.DataFrame:
        """Limpia columnas técnicas, imputa nulos y persiste el punto de control."""
        df_final = df_final.drop(columns=['fecha_cruce_mensual'])

        cols_clima = [DatasetKeys.TEMP_MEDIA, DatasetKeys.PRECIPITACION] 
        for col in cols_clima:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(df_final[col].mean())
                
        # Registro del checkpoint
        logger.info(f"Registrando dataset intermedio climático en {Paths.PROC_CSV_STEP_AEMET}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA] + [c for c in cols_clima if c in df_final.columns]
        df_final[cols_to_save].drop_duplicates().to_csv(Paths.PROC_CSV_STEP_AEMET, index=False)

        return df_final