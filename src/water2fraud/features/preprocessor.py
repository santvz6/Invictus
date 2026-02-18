import pandas as pd
from src.config import get_logger, DataSchema

logger = get_logger(__name__)

class WaterPreprocessor:
    @staticmethod
    def create_feature_matrix(df):
        logger.info("Transformando datos brutos a matriz de características...")
        
        df = WaterPreprocessor._process_timestamp(df)

        # --- Perfil horario medio ---
        hourly = df.pivot_table(
            index=DataSchema.ID_CONTADOR, 
            columns=DataSchema.HORA, 
            values=DataSchema.CONSUMO, 
            aggfunc="mean"
        ).fillna(0)
        hourly.columns = [f"H{int(c)}" for c in hourly.columns]

        # --- Perfil fin de semana ---
        periodic = df.groupby([DataSchema.ID_CONTADOR, DataSchema.ES_FINDE])[DataSchema.CONSUMO].mean().unstack().fillna(0)

        # Aseguramos que existan ambas columnas (True/False)
        for col in [True, False]:
            if col not in periodic.columns: periodic[col] = 0
        
        ratio_weekend = periodic[True] / (periodic[False] + 1e-6)
        
        # --- Estadísticos globales ---
        stats = df.groupby(DataSchema.ID_CONTADOR)[DataSchema.CONSUMO].agg(["mean", "std"]).fillna(0)
        # Renombramos para coincidir con el Schema
        stats.columns = [DataSchema.MEAN_CONSUMO, DataSchema.STD_CONSUMO]

        # Concatenación final
        X = pd.concat([hourly, ratio_weekend.rename(DataSchema.RATIO_WEEKEND), stats], axis=1)
        
        logger.info(f"Matriz generada con éxito. Dimensiones: {X.shape}")
        return X
    
    @staticmethod
    def _process_timestamp(df):
        """Convierte strings a datetime y extrae hora y día de la semana"""

        df = df.copy()
        df[DataSchema.TIMESTAMP] = pd.to_datetime(df[DataSchema.TIMESTAMP])
        
        # Hour
        df[DataSchema.HORA] = df[DataSchema.TIMESTAMP].dt.hour
        # Weekend
        df[DataSchema.ES_FINDE] = df[DataSchema.TIMESTAMP].dt.weekday >= 5
        
        return df