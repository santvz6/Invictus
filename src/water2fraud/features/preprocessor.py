import pandas as pd
from src.config import get_logger, DatasetKeys

logger = get_logger(__name__)

class WaterPreprocessor:
    @staticmethod
    def create_feature_matrix(df):
        logger.info("Transformando datos brutos a matriz de características...")
        
        df = WaterPreprocessor._process_dataframe(df)

        # --- Perfil  Consumo Medio ---
        hourly = df.pivot_table(
            index=DatasetKeys.ID_CONTADOR, 
            columns=DatasetKeys.HORA, 
            values=DatasetKeys.CONSUMO, 
            aggfunc="mean"
        ).fillna(0)
        hourly.columns = [f"H{int(c)}" for c in hourly.columns]

        # --- Perfil fin de semana ---
        periodic = df.groupby([DatasetKeys.ID_CONTADOR, DatasetKeys.ES_FINDE])[DatasetKeys.CONSUMO].mean().unstack().fillna(0)

        # Aseguramos que existan ambas columnas (True/False)
        for col in [True, False]:
            if col not in periodic.columns: periodic[col] = 0
        
        ratio_weekend = periodic[True] / (periodic[False] + 1e-6)
        
        # --- Estadísticos globales ---
        stats = df.groupby(DatasetKeys.ID_CONTADOR)[DatasetKeys.CONSUMO].agg(["mean", "std"]).fillna(0)
        # Renombramos para coincidir con el Schema
        stats.columns = [DatasetKeys.MEAN_CONSUMO, DatasetKeys.STD_CONSUMO]

        # Concatenación final
        X = pd.concat([hourly, ratio_weekend.rename(DatasetKeys.RATIO_WEEKEND), stats], axis=1)
        
        logger.info(f"Matriz generada con éxito. Dimensiones: {X.shape}")
        return X
    

    @staticmethod
    def _process_NaN(df):
        return df.copy().dropna()
    
    @staticmethod
    def _rename_df(df):
        return df.copy().rename(columns={
            "Barrio": DatasetKeys.BARRIO, 
            "Uso": DatasetKeys.USO, 
            "Fecha (aaaa/mm/dd)": DatasetKeys.FECHA,
            "Consumo (litros)": DatasetKeys.CONSUMO,
            "Nº Contratos" : DatasetKeys.NUM_CONTRATOS
        })

    @staticmethod
    def _process_dataframe(df):
        """"""
        df = df.copy()

        df = WaterPreprocessor._rename_df(df)
        df = WaterPreprocessor._process_NaN(df)

        # StrToInt
        for key in [DatasetKeys.CONSUMO, DatasetKeys.NUM_CONTRATOS]:
            df[key] = df[key].str.replace(",", "").astype(int)
        df[DatasetKeys.CONTRATO_RATIO] = df[DatasetKeys.CONSUMO] / df[DatasetKeys.NUM_CONTRATOS]

        # StrToDatetime
        df[DatasetKeys.FECHA] = pd.to_datetime(df[DatasetKeys.FECHA], format="%Y/%m/%d")
        df[DatasetKeys.MES] = df[DatasetKeys.FECHA].dt.month
        df[DatasetKeys.ES_FINDE] = df[DatasetKeys.FECHA].dt.weekday >= 5
        
        return df