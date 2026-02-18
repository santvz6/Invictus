import pandas as pd
from sklearn.ensemble import IsolationForest
from src.config import DataSchema, BusinessLabels, FraudStatus, AIConstants, get_logger

logger = get_logger(__name__)

class FraudDetector:
    """
    FASE 3 y 4: ...
    """
    def __init__(self, contamination=0.05):
        # El contamination es el % estimado de fraudes "extremos"
        self.iso_forest = IsolationForest(
            contamination=contamination, 
            random_state=AIConstants.RANDOM_STATE
        )

    def _logic_check(self, row, col_label_ia, col_contrato):
        """
        Lógica interna de la Fase 3. 
        Compara lo que hace el contador con lo que dice el papel.
        """
        ia = row[col_label_ia]
        contrato = row[col_contrato]

        if ia == BusinessLabels.TURISTICO and contrato == FraudStatus.CONTRATO_DOMESTICO:
            return FraudStatus.SOSPECHA_TURISTICO
        if ia == BusinessLabels.INDUSTRIAL_FUGA and contrato == FraudStatus.CONTRATO_DOMESTICO:
            return FraudStatus.ALERTA_TECNICA
        
        return FraudStatus.OK

    def run_detection_pipeline(self, df_results, col_label_ia, col_contrato, feature_cols) -> pd.DataFrame:
        """
        Ejecuta la FASE 3 y FASE 4 secuencialmente.
        """
        logger.info("Iniciando Fase 3: Cruce lógico de contratos...")
        
        # 1. Aplicamos la lógica de discrepancia (Fase 3)
        df_results[DataSchema.STATUS] = df_results.apply(
            self._logic_check, axis=1, args=(col_label_ia, col_contrato)
        )

        # 2. Filtramos solo los sospechosos
        df_suspects = df_results[df_results[DataSchema.STATUS] != FraudStatus.OK].copy()
        
        if df_suspects.empty:
            logger.warning("No se han detectado discrepancias iniciales.")
            return df_suspects

        logger.info(f"Fase 3 completada. {len(df_suspects)} sospechosos identificados.")

        # 3. Calculamos Score de Confianza (Fase 4)
        logger.info("Iniciando Fase 4: Cálculo de Anomaly Score con Isolation Forest...")
        
        # El modelo entrena SOLO con los sospechosos para ver quién destaca
        self.iso_forest.fit(df_suspects[feature_cols])
        
        # decision_function: valores bajos = más anómalo
        scores = self.iso_forest.decision_function(df_suspects[feature_cols])
        
        # Transformamos el score para que sea intuitivo (0 a 1, donde 1 es fraude seguro)
        # Invertimos y normalizamos el score de decisión
        df_suspects[DataSchema.CONFIDENCE] = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

        logger.info("Pipeline de detección finalizado.")
        return df_suspects.sort_values(by=DataSchema.CONFIDENCE, ascending=False)