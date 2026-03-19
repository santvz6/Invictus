import numpy as np
import pandas as pd

from src.config import get_logger, DatasetKeys, Paths
logger = get_logger(__name__)


class AMAEMProcessor:
   
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
        df[DatasetKeys.CONSUMO_RATIO] = df[DatasetKeys.CONSUMO] / df[DatasetKeys.NUM_CONTRATOS]

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


    def process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = AMAEMProcessor._rename_df(df)
        df = AMAEMProcessor._process_NaN(df)
        df = AMAEMProcessor._convert_dtype(df)
        df = AMAEMProcessor._one_hot_encoding(df)
        return df