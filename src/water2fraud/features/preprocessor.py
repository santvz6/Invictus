import numpy as np
import pandas as pd

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
        df = df.sort_values(by=[DatasetKeys.BARRIO, DatasetKeys.FECHA])
        
        # Seleccionamos las features que irán a la red neuronal
        feature_cols = [
            DatasetKeys.CONTRATO_RATIO,
            DatasetKeys.MES,
            # Aquí irían tus variables físicas / AEMET añadidas previamente al DF:
            # DatasetKeys.CONSUMO_FISICO_ESPERADO, 
            # 'temperatura_media', 'precipitacion', etc.
        ]
        
        # Asegurar que existan las columnas OHE, si no, rellenar con 0
        feature_cols = [DatasetKeys.USO_DOMESTICO, DatasetKeys.USO_COMERCIAL, DatasetKeys.USO_NO_DOMESTICO]
        for col in feature_cols:
            if col not in df.columns:
                # TODO: Mensaje
                raise ValueError("")
        
        sequences = []
        metadata = [] # Guardaremos a quién pertenece cada secuencia para identificar anomalías luego
        
        # Agrupamos por Barrio (y por tipo de uso si no hemos hecho variables unificadas)
        # Asumiendo que queremos una serie por cada par (Barrio, Uso)
        for barrio_id, group in df.groupby([DatasetKeys.BARRIO]):
            group_data = group[feature_cols].values
            
            # Ventana deslizante
            for i in range(len(group_data) - sequence_length + 1):
                seq = group_data[i : i + sequence_length]
                sequences.append(seq)
                # Guardamos info de la última fecha de la secuencia
                last_row = group.iloc[i + sequence_length - 1]
                metadata.append({
                    DatasetKeys.BARRIO: last_row[DatasetKeys.BARRIO],
                    DatasetKeys.FECHA: last_row[DatasetKeys.FECHA]
                })
                
        X = np.array(sequences) # Forma: (num_muestras, seq_length, num_features)
        meta_df = pd.DataFrame(metadata)
        
        logger.info(f"Generadas {X.shape[0]} secuencias de forma {X.shape[1]}x{X.shape[2]}")
        return X, meta_df
    

    ########################################### DATAFRAME PROCESSING
    @staticmethod
    def _rename_df(df: pd.DataFrame) -> pd.DataFrame:
        """Estandariza los nombres de las columnas basándose en las constantes de DatasetKeys."""
        return df.copy().rename(columns={
            "Barrio": DatasetKeys.BARRIO, 
            "Uso": DatasetKeys.USO, 
            "Fecha (aaaa/mm/dd)": DatasetKeys.FECHA,
            "Consumo (litros)": DatasetKeys.CONSUMO,
            "Nº Contratos" : DatasetKeys.NUM_CONTRATOS
        })


    @staticmethod
    def _process_NaN(df: pd.DataFrame) -> pd.DataFrame:
        """Elimina las filas que contienen valores nulos (NaN) del DataFrame."""
        return df.copy().dropna() # Eliminamos todos los nulos (al no representar gran parte de nuestros datos)
    
    @staticmethod
    def _convert_dtype(df: pd.DataFrame) -> pd.DataFrame:
        """ Convierte y ajusta los tipos de datos de las columnas numéricas y de fecha,
        y genera la característica derivada de ratio de consumo por contrato."""
        df = df.copy()

        # StrToInt
        for key in [DatasetKeys.CONSUMO, DatasetKeys.NUM_CONTRATOS]:
            df[key] = df[key].str.replace(",", "").astype(int)
        df[DatasetKeys.CONTRATO_RATIO] = df[DatasetKeys.CONSUMO] / df[DatasetKeys.NUM_CONTRATOS]

        # StrToDatetime
        df[DatasetKeys.FECHA] = pd.to_datetime(df[DatasetKeys.FECHA], format="%Y/%m/%d")
        df[DatasetKeys.MES] = df[DatasetKeys.FECHA].dt.month
        return df
    
    @staticmethod
    def _one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica One-Hot Encoding a la variable categórica de Uso del agua."""
        df = df.copy()
        dummies = pd.get_dummies(df[DatasetKeys.USO], prefix=DatasetKeys.USO, dtype=int)
        return  pd.concat([df, dummies], axis=1)


    @staticmethod
    def _save_processed_df(df: pd.DataFrame) -> pd.DataFrame:
        """Persiste el DataFrame preprocesado en disco en formato CSV."""
        df.to_csv(Paths.PROC_CSV_AMAEM, index=False)
        
    @staticmethod
    def process_raw_data(df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline principal de limpieza que orquesta los pasos de preprocesamiento de un DataFrame crudo."""
        df = df.copy()
        df = WaterPreprocessor._rename_df(df)
        df = WaterPreprocessor._process_NaN(df)
        df = WaterPreprocessor._convert_dtype(df)
        df = WaterPreprocessor._one_hot_encoding(df)
        WaterPreprocessor._save_processed_df(df)
        return df