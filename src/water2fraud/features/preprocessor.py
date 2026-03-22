import numpy as np
import pandas as pd

from src.water2fraud.features.amaem_processor import AMAEMProcessor
from src.water2fraud.features.ine_tourism_processor import INETourismProcessor
from src.water2fraud.features.aemet_processor import AEMETProcessor
from src.water2fraud.features.fisicos_processor import FisicosProcessor
from src.water2fraud.features.sentinel_processor import SentinelProcessor
from src.water2fraud.features.gva_processor import GVAProcessor
from src.config import get_logger, DatasetKeys, Paths
logger = get_logger(__name__)


class WaterPreprocessor:
    """
    Módulo encargado de la limpieza, transformación y estructuración de los datos
    crudos de agua hacia un formato ingerible por las redes neuronales temporales.
    """

    @staticmethod
    def create_sequences(df: pd.DataFrame, sequence_length=12) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Convierte el DataFrame tabular en secuencias temporales 3D (ventanas deslizantes)
        para alimentar el LSTM-AE.
        
        Agrupa los datos por barrio y extrae ventanas cronológicas, reteniendo las
        variables predictoras y generando un registro paralelo de metadatos.

        Args:
            df (pd.DataFrame): DataFrame preprocesado y cronológicamente ordenado.
            sequence_length (int, optional): Tamaño de la ventana de meses a generar. Por defecto es 12.

        Returns:
            tuple:
                - X (np.ndarray): Tensor 3D de secuencias de forma (num_muestras, seq_length, num_features).
                - meta_df (pd.DataFrame): Metadatos correspondientes al último mes de cada secuencia extraída.
                
        Raises:
            ValueError: Si alguna columna requerida (One-Hot Encoding) no se encuentra en el DataFrame.
        """
        logger.info(f"Generando secuencias temporales de tamaño {sequence_length} meses...")
        
        # Ordenamos cronológicamente
        df = df.sort_values(by=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA])
        
        # Seleccionamos las features que irán a la red neuronal
        feature_cols = [
            # AMAEM
            DatasetKeys.CONSUMO_RATIO,
            DatasetKeys.MES_SIN,
            DatasetKeys.MES_COS,

            # INE - TOURISM
            DatasetKeys.NUM_VT_BARRIO_INE,     
            DatasetKeys.PCT_VT_BARRIO_INE,
            DatasetKeys.OCUP_VT_PROV_INE,    
            DatasetKeys.PERNOCT_VT_PROV_INE,  

            # FÍSICOS
            DatasetKeys.CONSUMO_FISICO_ESPERADO, 

            # AEMET
            DatasetKeys.TEMP_MEDIA, 
            DatasetKeys.PRECIPITACION,
            
            # SENTINEL
            DatasetKeys.NDVI_SATELITE,
            
            # GVA
            DatasetKeys.NUM_VT_BARRIO_GVA,
            DatasetKeys.PLAZAS_VIVIENDAS_GVA,
            DatasetKeys.NUM_HOTELES_BARRIO_GVA,
            DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA,
            
            # ENGINEERED FEATURES
            DatasetKeys.NUM_VT_ILEGALES
        ]
        
        # Asegurar que solo usamos características que existan realmente en el DataFrame
        # (protege el código si algún archivo externo como Sentinel no se encontró)
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Asegurar que existan las columnas OHE, si no, rellenar con 0
        #ohe_cols = [DatasetKeys.USO_DOMESTICO, DatasetKeys.USO_COMERCIAL, DatasetKeys.USO_NO_DOMESTICO]
        #for col in ohe_cols:
        #    if col not in df.columns:
        #        raise ValueError(f"No se ha aplicado One Hot Encoding | Columnas df: {df.columns}")
        #
        #feature_cols.extend(ohe_cols)

        sequences = []
        metadata = [] # Guardaremos a quién pertenece cada secuencia para identificar anomalías luego
        
        # Agrupamos por Barrio y por tipo de USO
        # Queremos una serie por cada par (Barrio, Uso)
        for (barrio, uso), group in df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO]):
            group_data = group[feature_cols].values
            
            # Ventana deslizante
            for i in range(len(group_data) - sequence_length + 1):
                seq = group_data[i : i + sequence_length]
                sequences.append(seq)
                # Guardamos info de la última fecha de la secuencia
                last_row = group.iloc[i + sequence_length - 1]
                metadata.append({
                    DatasetKeys.BARRIO: last_row[DatasetKeys.BARRIO],
                    DatasetKeys.USO: last_row[DatasetKeys.USO],
                    DatasetKeys.FECHA: last_row[DatasetKeys.FECHA]
                })
                
        X = np.array(sequences) # Forma: (num_muestras, seq_length, num_features)
        meta_df = pd.DataFrame(metadata)
        
        logger.info(f"Generadas {X.shape[0]} secuencias de forma {X.shape[1]}x{X.shape[2]}")
        return X, meta_df, feature_cols
    

    ########################################### DATAFRAME PROCESSING
    @staticmethod
    def _scale_features(df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import MinMaxScaler
        """
        Escala las variables numéricas continuas al rango [0, 1] para evitar 
        la saturación de gradientes en la red neuronal LSTM.
        """
        df = df.copy()
        scaler = MinMaxScaler()
        
        # Aquí añadiremos en el futuro la temperatura, precipitaciones, consumo físico, etc.
        # ? Realmente queremos MinMaxScaler para todos los features 
        # ? o para algunos es mejor StandardScaler
        cols_to_scale = [
            DatasetKeys.CONSUMO_RATIO,
            DatasetKeys.NUM_VT_BARRIO_INE,
            DatasetKeys.PCT_VT_BARRIO_INE,
            DatasetKeys.OCUP_VT_PROV_INE,
            DatasetKeys.PERNOCT_VT_PROV_INE,
            DatasetKeys.TEMP_MEDIA,
            DatasetKeys.PRECIPITACION,
            DatasetKeys.CONSUMO_FISICO_ESPERADO,
            DatasetKeys.NDVI_SATELITE,
            DatasetKeys.NUM_VT_BARRIO_GVA,
            DatasetKeys.PLAZAS_VIVIENDAS_GVA,
            DatasetKeys.NUM_HOTELES_BARRIO_GVA,
            DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA,
            DatasetKeys.NUM_VT_ILEGALES
        ]
        
        # Filtramos por seguridad
        cols_present = [c for c in cols_to_scale if c in df.columns]
        if cols_present:
            # ! A pesar de que hay 'Data Leakage'
            # ! Hay que recordar que estamos entrenando un Autoencoder
            df[cols_present] = scaler.fit_transform(df[cols_present])
        
        # Escalado del mes
        meses = df[DatasetKeys.MES]
        df[DatasetKeys.MES_SIN] = (np.sin(2 * np.pi * meses / 12) + 1) / 2
        df[DatasetKeys.MES_COS] = (np.cos(2 * np.pi * meses / 12) + 1) / 2

        return df

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características derivadas complejas cruzando múltiples fuentes.
        Reparte el total de la provincia registrado en la GVA hacia los barrios
        usando las ponderaciones del INE y estima el gap de viviendas ilegales.
        """
        logger.info("Calculando variables derivadas (Reparto ponderado GVA e Ilegalidad)...")
        df = df.copy()

        # 1. Obtenemos una fila única por Barrio y Fecha para no duplicar sumas al iterar por Usos
        df_unique = df.drop_duplicates(subset=[DatasetKeys.FECHA, DatasetKeys.BARRIO]).copy()
        
        # 2. Calculamos el total de VT del INE por mes para toda la ciudad
        total_ine_mes = df_unique.groupby(DatasetKeys.FECHA)[DatasetKeys.NUM_VT_BARRIO_INE].transform('sum')
        
        # 3. Calculamos la cuota (porcentaje) que le corresponde a cada barrio
        df_unique['pct_barrio'] = df_unique[DatasetKeys.NUM_VT_BARRIO_INE] / total_ine_mes.replace(0, 1)
        
        # 4. Traemos el porcentaje de vuelta al dataframe principal
        df = pd.merge(df, df_unique[[DatasetKeys.FECHA, DatasetKeys.BARRIO, 'pct_barrio']], 
                      on=[DatasetKeys.FECHA, DatasetKeys.BARRIO], how='left')

        # 5. Distribuimos los totales de la GVA según la cuota de cada barrio
        cols_gva = [
            DatasetKeys.NUM_VT_BARRIO_GVA,
            DatasetKeys.PLAZAS_VIVIENDAS_GVA,
            DatasetKeys.NUM_HOTELES_BARRIO_GVA,
            DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA
        ]
        
        for col in cols_gva:
            if col in df.columns:
                df[col] = (df[col] * df['pct_barrio']).round(2)

        # 6. Calcular Viviendas Ilegales (Estimación INE - Registros Oficiales GVA ponderados)
        df[DatasetKeys.NUM_VT_ILEGALES] = (df[DatasetKeys.NUM_VT_BARRIO_INE] - df[DatasetKeys.NUM_VT_BARRIO_GVA]).clip(lower=0)

        df = df.drop(columns=['pct_barrio'])
        return df
    
    @staticmethod
    def _save_processed_df(df: pd.DataFrame, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """Persiste el DataFrame preprocesado en disco en formato CSV."""
        df.to_csv(Paths.PROC_CSV_AMAEM_NOT_SCALED, index=False)
        df_scaled.to_csv(Paths.PROC_CSV_AMAEM_SCALED, index=False)
    
    @staticmethod
    def process_raw_data(df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline principal de limpieza que orquesta los pasos de preprocesamiento de un DataFrame crudo."""
        df = df.copy()
        
        df = AMAEMProcessor.process(df)
        # Turismo
        df = INETourismProcessor.process(df)
        df = GVAProcessor.process(df)
        # Clima
        df = AEMETProcessor.process(df)
        df = SentinelProcessor.process(df)
        df = FisicosProcessor.process(df)

        df = WaterPreprocessor._engineer_features(df)

        df_scaled = WaterPreprocessor._scale_features(df)
        WaterPreprocessor._save_processed_df(df, df_scaled)
        return df_scaled