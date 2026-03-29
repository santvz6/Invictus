"""
Módulo Orquestador de Preprocesamiento y Preparación de Tensores.

Este componente actúa como el cerebro de la etapa de features, coordinando la 
limpieza secuencial de múltiples fuentes (AMAEM, INE, GVA, AEMET, Sentinel) y 
transformando los datos tabulares en secuencias temporales 3D aptas para modelos 
de Deep Learning (LSTM Autoencoders).
"""

import pandas as pd
import numpy as np

from src.features.ine_tourism_processor import INETourismProcessor
from src.features.sentinel_processor import SentinelProcessor
from src.features.aemet_processor import AEMETProcessor
from src.features.amaem_processor import AMAEMProcessor
from src.features.gva_processor import GVAProcessor
from src.features.holiday_barrio_processor import HolidayBarrioProcessor
from src.config import get_logger, DatasetKeys, Paths

# Logger central del pipeline de características
logger = get_logger(__name__)


class WaterPreprocessor:
    """
    Orquestador global de transformación de datos para detección de fraude.
    
    Centraliza la configuración de variables predictoras, su normalización estocástica 
    y la generación de ventanas deslizantes (sliding windows) para el aprendizaje temporal.
    """
    
    # Constantes de tipado de escalado
    MIN_MAX = "min-max"
    ROBUST  = "robust"
    SIN_COS = "sin-cos"

    # Diccionario maestro de características predictoras (Features)
    # Define cómo debe ser tratada cada variable antes de entrar en la red neuronal.
    FEATURES = {
        # AMAEM
        DatasetKeys.CONSUMO_RATIO: ROBUST,
        DatasetKeys.MES_SIN: SIN_COS,    
        DatasetKeys.MES_COS: SIN_COS,

        # INE - TOURISM
        # DatasetKeys.NUM_VT_BARRIO_INE: MIN_MAX,           # Eliminado por redundancia con el PCT_VT_SIN_REGISTRAR
        # DatasetKeys.PCT_VT_BARRIO_INE: MIN_MAX,           # Eliminado por redundancia con el PCT_VT_SIN_REGISTRAR
        # DatasetKeys.OCUP_VT_PROV_INE: MIN_MAX,            # Eliminado (información por Provincias)
        # DatasetKeys.PERNOCT_VT_PROV_INE: MIN_MAX,         # Eliminado (información por Provincias)

        # AEMET
        DatasetKeys.TEMP_MEDIA: MIN_MAX, 
        DatasetKeys.PRECIPITACION: MIN_MAX,
        
        # SENTINEL
        DatasetKeys.NDVI_SATELITE: MIN_MAX,
        
        # GVA
        # DatasetKeys.NUM_VT_BARRIO_GVA: MIN_MAX,           # Eliminado por redundancia con el PCT_VT_SIN_REGISTRAR
        # DatasetKeys.PLAZAS_VIVIENDAS_GVA: MIN_MAX,        # Eliminado por redundancia con el PCT_VT_SIN_REGISTRAR
        # DatasetKeys.NUM_HOTELES_BARRIO_GVA: MIN_MAX,      # Optamos por utilizar Plazas Hoteles
        DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA: MIN_MAX,
        
        # FESTIVOS
        #DatasetKeys.DIAS_FESTIVOS: MIN_MAX,                # ! Eliminado porque no es feature de predicción

        # ENGINEERED FEATURES
        # DatasetKeys.NUM_VT_SIN_REGISTRAR: MIN_MAX,         # Eliminado por redundancia con el PCT_VT_SIN_REGISTRAR
        # DatasetKeys.PCT_VT_SIN_REGISTRAR: ROBUST           # ! Eliminado porque no es feature de predicción
    }

    @staticmethod
    def _load_data() -> pd.DataFrame:
        input_path = Paths.RAW_CSV_AMAEM
        if not input_path.exists():
            logger.error(f"Error crítico: No se encuentra el archivo en {input_path}")
            raise FileNotFoundError(f"Error crítico: No se encuentra el archivo en {input_path}")
        logger.info(f"Cargando datos desde {input_path}...")
        return pd.read_csv(input_path)
    

    @staticmethod
    def _scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Normaliza las magnitudes de las variables para optimizar el aprendizaje profundo.
        
        Utiliza MinMaxScaler para comprimir variables continuas y transformaciones 
        cíclicas (sin/cos) para variables periódicas como el mes.
        """
        from sklearn.preprocessing import MinMaxScaler, RobustScaler
        df = df.copy()
        scalers = {}

        for col, scale_type in WaterPreprocessor.FEATURES.items():
            if col in df.columns:
                if scale_type == WaterPreprocessor.ROBUST:
                    scaler = RobustScaler()
                    
                    df[col] = scaler.fit_transform(df[[col]])
                    scalers[col] = scaler
                elif scale_type == WaterPreprocessor.MIN_MAX:
                    scaler = MinMaxScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    scalers[col] = scaler
        
        # Codificación cíclica del tiempo (Meses)
        # Permite que la red entienda que diciembre (12) está cerca de enero (1)
        meses = df[DatasetKeys.MES]
        df[DatasetKeys.MES_SIN] = (np.sin(2 * np.pi * meses / 12) + 1) / 2
        df[DatasetKeys.MES_COS] = (np.cos(2 * np.pi * meses / 12) + 1) / 2

        return df, scalers


    @staticmethod
    def _INE_GVA_gap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta lógica de ingeniería de características complejas basada en el 'Gap' de legalidad.
        
        Calcula la discrepancia entre el turismo reportado (INE) y el registrado (GVA), 
        distribuyendo pesos municipales hacia el detalle de barrio.
        """
        df = df.copy()

        # Deduplicación para cálculo de cuotas geográficas puras
        df_unique = df.drop_duplicates(subset=[DatasetKeys.FECHA, DatasetKeys.BARRIO]).copy()
        
        # 1. Totalización de la huella turística de la ciudad por mes (según INE)
        total_ine_mes = df_unique.groupby(DatasetKeys.FECHA)[DatasetKeys.NUM_VT_BARRIO_INE].transform('sum')
        
        # 2. Determinación del peso específico de cada barrio en la carga turística total
        df_unique['pct_barrio'] = df_unique[DatasetKeys.NUM_VT_BARRIO_INE] / total_ine_mes.replace(0, 1)
        
        # 3. Propagación de pesos al dataset principal
        df = pd.merge(df, df_unique[[DatasetKeys.FECHA, DatasetKeys.BARRIO, 'pct_barrio']], 
                      on=[DatasetKeys.FECHA, DatasetKeys.BARRIO], how='left')

        # 4. Distribución de métricas oficiales (GVA) basada en la cuota ponderada
        cols_gva = [
            DatasetKeys.NUM_VT_BARRIO_GVA,
            DatasetKeys.PLAZAS_VIVIENDAS_GVA,
            DatasetKeys.NUM_HOTELES_BARRIO_GVA,
            DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA
        ]
        
        for col in cols_gva:
            if col in df.columns:
                df[col] = (df[col] * df['pct_barrio']).round(2)

        # 5. ESTIMACIÓN DE ILEGALIDAD:
        # Delta entre el total estimado por INE y el registro oficial de la GVA ponderado.
        # Un valor positivo indica presencia probable de pisos turísticos no declarados.
        df[DatasetKeys.NUM_VT_SIN_REGISTRAR] = (df[DatasetKeys.NUM_VT_BARRIO_INE] - df[DatasetKeys.NUM_VT_BARRIO_GVA]).clip(lower=0)
        
        # Normalización por densidad de contratos
        df[DatasetKeys.PCT_VT_SIN_REGISTRAR] = ((df[DatasetKeys.NUM_VT_SIN_REGISTRAR] / df[DatasetKeys.NUM_CONTRATOS].replace(0, np.nan)) * 100).fillna(0).round(2)

        df = df.drop(columns=['pct_barrio']) 
        return df
    
    @staticmethod
    def _save_processed_df(df_not_scaled: pd.DataFrame, df_scaled: pd.DataFrame) -> None:
        """Centraliza la persistencia de los diferentes estados del dataset."""
        df_not_scaled.to_csv(Paths.PROC_CSV_AMAEM_NOT_SCALED, index=False)
        df_scaled.to_csv(Paths.PROC_CSV_AMAEM_SCALED, index=False)
    

    @staticmethod
    def process_all_data() -> tuple[pd.DataFrame, dict]:
        """
        ...
        """
        df_not_scaled = WaterPreprocessor._load_data()
        
        # Fase A: Ingesta y limpieza base (Agua)
        df_not_scaled = AMAEMProcessor.process(df_not_scaled)
        
        # Fase B: Enriquecimiento Turístico (INE y GVA)
        df_not_scaled = INETourismProcessor.process(df_not_scaled)
        df_not_scaled = GVAProcessor.process(df_not_scaled)
        
        # Fase C: Enriquecimiento Ambiental (AEMET y Sentinel)
        df_not_scaled = AEMETProcessor.process(df_not_scaled)
        df_not_scaled = SentinelProcessor.process(df_not_scaled)
        
        # Fase D: Enriquecimiento de Festivos
        df_not_scaled = HolidayBarrioProcessor.process(df_not_scaled)

        # Fase E: Ingeniería de Variables de Fraude (Gap de Ilegalidad)
        df_not_scaled = WaterPreprocessor._INE_GVA_gap(df_not_scaled)

        # Fase E: Normalización para Deep Learning
        df_scaled, scalers = WaterPreprocessor._scale_features(df_not_scaled)
        
        # Eliminamos columnas NO utilizadas
        allways_used = [DatasetKeys.FECHA, DatasetKeys.BARRIO, DatasetKeys.NUM_CONTRATOS, DatasetKeys.USO]
        df_scaled = df_scaled[allways_used + list(WaterPreprocessor.FEATURES.keys())]

        not_scaled_columns = [c for c in list(WaterPreprocessor.FEATURES.keys()) if c in df_not_scaled.columns]
        df_not_scaled      = df_not_scaled[allways_used + not_scaled_columns]

        # Persistencia del estado final
        WaterPreprocessor._save_processed_df(df_not_scaled, df_scaled)

        return df_scaled, df_not_scaled, scalers