"""
Módulo de ingesta y preprocesamiento base de datos de AMAEM.

Este componente actúa como la primera capa de limpieza del pipeline, encargándose 
de la estandarización de cabeceras, saneamiento de valores nulos y normalización 
de tipos de datos para asegurar la consistencia en las etapas posteriores.
"""

import numpy as np
import pandas as pd

from src.config import get_logger, DatasetKeys, Paths

# Logger configurado para la trazabilidad de la ingesta inicial
logger = get_logger(__name__)

class AMAEMProcessor:
    """
    Procesador de limpieza y normalización de registros de facturación de agua.
    
    Responsable de transformar los datos en bruto (Raw) en un formato estructurado 
    y tipificado apto para el enriquecimiento con fuentes externas.
    """
   
    @staticmethod
    def _rename_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza los nombres de las columnas basándose en las constantes de DatasetKeys.
        Garantiza que el resto del sistema sea agnóstico a cambios en las cabeceras del CSV de entrada.
        """
        return df.copy().rename(columns={
            "Barrio": DatasetKeys.BARRIO, 
            "Uso": DatasetKeys.USO, 
            "Fecha (aaaa/mm/dd)": DatasetKeys.FECHA,
            "Consumo (litros)": DatasetKeys.CONSUMO,
            "Nº Contratos" : DatasetKeys.NUM_CONTRATOS
        })

    @staticmethod
    def _process_NaN(df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros con valores incompletos.
        Se opta por la eliminación total dado que el volumen de nulos es residual 
        y no compromete la representatividad estadística de los barrios.
        """
        return df.copy().dropna()
    
    @staticmethod
    def _convert_dtype(df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Normaliza los tipos de datos y genera métricas derivadas fundamentales.
        
        Realiza:
        1. Limpieza de separadores de miles y conversión a enteros.
        2. Cálculo del Ratio de Consumo (Litros/Contrato), variable objetivo del modelo.
        3. Parseo de fechas y extracción del componente mensual para análisis estacional.
        """
        df = df.copy()

        # Limpieza de strings numéricos (ej: '1,000' -> 1000)
        for key in [DatasetKeys.CONSUMO, DatasetKeys.NUM_CONTRATOS]:
            if df[key].dtype == 'object':
                df[key] = df[key].str.replace(",", "").astype(int)
        
        # Ingeniería de características base: Consumo promedio por unidad alojativa/contrato
        df[DatasetKeys.CONSUMO_RATIO] = df[DatasetKeys.CONSUMO] / df[DatasetKeys.NUM_CONTRATOS]

        # Estandarización temporal
        df[DatasetKeys.FECHA] = pd.to_datetime(df[DatasetKeys.FECHA], format="%Y/%m/%d")
        df[DatasetKeys.MES] = df[DatasetKeys.FECHA].dt.month
        return df

    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline de limpieza inicial de datos de AMAEM.

        Args:
            df (pd.DataFrame): DataFrame original cargado desde CSV raw.

        Returns:
            pd.DataFrame: DataFrame limpio, tipificado y con métricas base calculadas.
        """
        logger.info("Iniciando preprocesamiento base de datos AMAEM...")
        df = df.copy()
        
        # Secuencia de limpieza y normalización
        df = AMAEMProcessor._rename_df(df)
        df = AMAEMProcessor._process_NaN(df)
        df = AMAEMProcessor._convert_dtype(df)
        
        # Registro del primer punto de control del pipeline
        logger.info(f"Registrando dataset intermedio AMAEM en {Paths.PROC_CSV_STEP_AMAEM}")
        df.to_csv(Paths.PROC_CSV_STEP_AMAEM, index=False)
        
        return df