"""
Módulo Orquestador de Preprocesamiento y Preparación de Tensores.

Este componente actúa como el cerebro de la etapa de features, coordinando la 
limpieza secuencial de múltiples fuentes (AMAEM, INE, GVA, AEMET, Sentinel) y 
transformando los datos tabulares en secuencias temporales 3D aptas para modelos 
de Deep Learning (LSTM Autoencoders).
"""

import pandas as pd
import numpy as np

from src.water2fraud.features.ine_tourism_processor import INETourismProcessor
from src.water2fraud.features.sentinel_processor import SentinelProcessor
from src.water2fraud.features.fisicos_processor import FisicosProcessor
from src.water2fraud.features.aemet_processor import AEMETProcessor
from src.water2fraud.features.amaem_processor import AMAEMProcessor
from src.water2fraud.features.gva_processor import GVAProcessor
from src.water2fraud.features.holiday_barrio_processor import HolidayBarrioProcessor
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
    def create_sequences(df: pd.DataFrame, sequence_length=12) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
        """
        Transforma datos tabulares en tensores 3D para redes recurrentes.
        
        Aplica un enfoque de ventana deslizante por cada segmento (Barrio, Uso), 
        asegurando que el modelo aprenda patrones secuenciales de consumo.

        Args:
            df (pd.DataFrame): DataFrame completamente preprocesado.
            sequence_length (int): Longitud de la memoria temporal (meses).

        Returns:
            tuple: (X (tensor 3D), meta_df (referencias de barrio/fecha), feature_cols (nombres)).
        """
        logger.info(f"Generando secuencias temporales de tamaño {sequence_length} meses...")
        
        # Ordenación rigurosa necesaria para la coherencia de la ventana temporal
        df = df.sort_values(by=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA])        
        
        # Selección dinámica de columnas presentes en el dataset
        feature_cols = [col for col in WaterPreprocessor.FEATURES if col in df.columns]

        sequences = []
        metadata = [] 
        
        # Iteración por clústers geográficos y de uso para evitar solapamientos entre barrios
        for (barrio, uso), group in df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO]):
            group_data = group[feature_cols].values
            
            # Construcción de ventanas deslizantes
            for i in range(len(group_data) - sequence_length + 1):
                seq = group_data[i : i + sequence_length]
                sequences.append(seq)
                
                # Anclaje de metadatos al último mes de la ventana (momento de la predicción)
                last_row = group.iloc[i + sequence_length - 1]
                metadata.append({
                    DatasetKeys.BARRIO: last_row[DatasetKeys.BARRIO],
                    DatasetKeys.USO: last_row[DatasetKeys.USO],
                    DatasetKeys.FECHA: last_row[DatasetKeys.FECHA]
                })
                
        X = np.array(sequences) 
        meta_df = pd.DataFrame(metadata)
        
        logger.info(f"Generadas {X.shape[0]} secuencias de forma {X.shape[1]}x{X.shape[2]}")
        return X, meta_df, feature_cols

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
                    
                    # Aplicar logaritmo natural a la variable de consumo
                    df[col] = np.log1p(df[col])
                    
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
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta lógica de ingeniería de características complejas basada en el 'Gap' de legalidad.
        
        Calcula la discrepancia entre el turismo reportado (INE) y el registrado (GVA), 
        distribuyendo pesos municipales hacia el detalle de barrio.
        """
        logger.info("Calculando variables derivadas (Reparto ponderado GVA e Ilegalidad)...")
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
    def process_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Orquestación maestra del pipeline de datos 'Water2Fraud'.
        
        Encadena secuencialmente todos los procesadores especializados y culmina con 
        la ingeniería de características y el escalado final.
        """
        df_not_scaled = df.copy()
        
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
        df_not_scaled = WaterPreprocessor._engineer_features(df_not_scaled)

        # Fase E: Normalización para Deep Learning
        df_scaled, scalers = WaterPreprocessor._scale_features(df_not_scaled)
        
        # Persistencia del estado final
        WaterPreprocessor._save_processed_df(df_not_scaled, df_scaled)

        return df_scaled, scalers