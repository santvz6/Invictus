import pandas as pd
from src.config import get_logger, DatasetKeys, Paths


logger = get_logger(__name__)

class WaterPreprocessor:
    @staticmethod
    def create_feature_matrix(df: pd.DataFrame):
        logger.info("Transformando datos brutos a matriz de características...")
        
        df = WaterPreprocessor.process_dataframe(df)
        X = None
        
        return X
    
    @staticmethod
    def _one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dummies = pd.get_dummies(df[DatasetKeys.USO], prefix=DatasetKeys.USO, dtype=int)
        return  pd.concat([df, dummies], axis=1)

    @staticmethod
    def _process_NaN(df: pd.DataFrame) -> pd.DataFrame:
        # Eliminamos todos los nulos (al no representar gran parte de nuestros datos)
        return df.copy().dropna()
    
    @staticmethod
    def _rename_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().rename(columns={
            "Barrio": DatasetKeys.BARRIO, 
            "Uso": DatasetKeys.USO, 
            "Fecha (aaaa/mm/dd)": DatasetKeys.FECHA,
            "Consumo (litros)": DatasetKeys.CONSUMO,
            "Nº Contratos" : DatasetKeys.NUM_CONTRATOS
        })

    def _convert_dtype(df: pd.DataFrame) -> pd.DataFrame:
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
    def _save_processed_df(df: pd.DataFrame) -> pd.DataFrame:
        df.to_csv(Paths.PROC_CSV_AMAEM, index=False)
        
    @staticmethod
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """"""
        df = df.copy()

        df = WaterPreprocessor._rename_df(df)
        df = WaterPreprocessor._process_NaN(df)
        df = WaterPreprocessor._convert_dtype(df)
        WaterPreprocessor._save_processed_df(df)
        return df