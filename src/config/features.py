"""
Módulo de Configuración Científica de Características (Features).

Centraliza la definición de las variables utilizadas en los distintos pipelines.
Actúa como 'Ground Truth' tanto para el preprocesamiento de Deep Learning 
como para la asignación física de causas de anomalías.
"""

from src.config.string_keys import DatasetKeys

class FeatureScaling:
    """Tipos de escalado permitidos para el motor de Deep Learning."""
    MIN_MAX = "min-max"
    ROBUST  = "robust"
    SIN_COS = "sin-cos"



class FeatureConfig:
    """
    Configuración central (Ground Truth) de features predictoras.

    Si se añade o comenta una variable aquí, tanto el Preprocesador 
    como el Modelo Físico se adaptarán automáticamente a su presencia o ausencia.
    """
    
    # 1. Pipeline de Deep Learning (Preprocesador)
    # Define qué columnas se utilizan y cómo deben escalarse
    PIPELINE_FEATURES = {
        # AMAEM
        DatasetKeys.CONSUMO_RATIO: FeatureScaling.ROBUST,
        DatasetKeys.MES_SIN: FeatureScaling.SIN_COS,    
        DatasetKeys.MES_COS: FeatureScaling.SIN_COS,

        # LLUVIA - SEQUIA (AEMET)
        DatasetKeys.TEMP_MEDIA:    FeatureScaling.MIN_MAX, 
        DatasetKeys.PRECIPITACION: FeatureScaling.MIN_MAX,
        
        # VEGETACION (SENTINEL)
        DatasetKeys.NDVI_SATELITE: FeatureScaling.MIN_MAX,
        
        # FESTIVOS
        DatasetKeys.PCT_FESTIVOS: FeatureScaling.MIN_MAX,

        # TURISMO (INE - GVA)
        DatasetKeys.PCT_VT_SIN_REGISTRAR: FeatureScaling.ROBUST

    }

    # 2. Triaje de Anomalías (Modelo Físico de 6 Niveles)
    # Define qué variables exógenas se analizarán para asignar el % de causa (peso/z-score)
    CAUSAS_EXOGENAS = {
        DatasetKeys.TEMP_MEDIA: DatasetKeys.PCT_CALOR_FRIO,
        DatasetKeys.PRECIPITACION: DatasetKeys.PCT_LLUVIA_SEQUIA,
        DatasetKeys.NDVI_SATELITE: DatasetKeys.PCT_VEGETACION,
        DatasetKeys.PCT_VT_SIN_REGISTRAR: DatasetKeys.PCT_TURISMO,
        DatasetKeys.PCT_FESTIVOS: DatasetKeys.PCT_FIESTA
    }
