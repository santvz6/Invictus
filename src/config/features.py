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

        # TURISMO - Presión Turística Real (INE/Provincial) [MEJORA 1]
        # Se eliminó PCT_VT_SIN_REGISTRAR (medía diferencial de oferta, no presión sobre consumo)
        DatasetKeys.OCUP_VT_PROV_INE:     FeatureScaling.MIN_MAX,   # Alojamientos turísticos ocupados
        DatasetKeys.PERNOCT_VT_PROV_INE:  FeatureScaling.ROBUST,    # Noches ocupadas (presión real)

        # FESTIVOS - Mejorados [MEJORA 2]
        # Se eliminó PCT_FESTIVOS (porcentaje anual, métrica débil)
        DatasetKeys.DIAS_FESTIVOS: FeatureScaling.MIN_MAX,   # Conteo directo de días festivos
        DatasetKeys.ES_PUENTE:     FeatureScaling.MIN_MAX,   # Binaria: ≥2 días = efecto puente

        # ESTACIONALIDAD BINARIA [MEJORA 4]
        # Ortogonales a Fourier → el RF puede aprender sus deltas sin interferencia
        DatasetKeys.SEMANA_SANTA: FeatureScaling.MIN_MAX,    # Marzo/Abril
        DatasetKeys.VERANO:       FeatureScaling.MIN_MAX,    # Junio/Julio/Agosto
        DatasetKeys.NAVIDAD:      FeatureScaling.MIN_MAX,    # Diciembre/Enero
    }

    # 2. Triaje de Anomalías (Modelo Físico de 6 Niveles)
    # Define qué variables exógenas se analizarán para asignar el % de causa (peso/z-score)
    # Múltiples features pueden apuntar a la misma causa → sus SHAP values se suman
    CAUSAS_EXOGENAS = {
        DatasetKeys.TEMP_MEDIA:           DatasetKeys.PCT_CALOR_FRIO,
        DatasetKeys.PRECIPITACION:        DatasetKeys.PCT_LLUVIA_SEQUIA,
        DatasetKeys.NDVI_SATELITE:        DatasetKeys.PCT_VEGETACION,
        # Turismo real (presión sobre consumo, no diferencial administrativo)
        DatasetKeys.OCUP_VT_PROV_INE:    DatasetKeys.PCT_TURISMO,
        DatasetKeys.PERNOCT_VT_PROV_INE: DatasetKeys.PCT_TURISMO,
        DatasetKeys.VERANO:              DatasetKeys.PCT_TURISMO,    # Presión vacacional
        # Festivos agrupados
        DatasetKeys.DIAS_FESTIVOS:       DatasetKeys.PCT_FIESTA,
        DatasetKeys.ES_PUENTE:           DatasetKeys.PCT_FIESTA,
        DatasetKeys.SEMANA_SANTA:        DatasetKeys.PCT_FIESTA,
        DatasetKeys.NAVIDAD:             DatasetKeys.PCT_FIESTA,
    }
