import pandas as pd
import argparse

from pathlib import Path
from datetime import datetime

from src.features.preprocessor import WaterPreprocessor
from src.model import ModeloFisico
from src.config import get_logger, Paths, FeatureConfig

Paths.init_project()
logger = get_logger(__name__)

class WaterApp:
    """
    Orquestador del pipeline físico 'Water2Fraud'.
    Se enfoca en modelado de Fourier e impacto de factores exógenos (Random Forest).
    """

    @staticmethod
    def run_pipeline() -> pd.DataFrame:
        """
        Ejecuta el pipeline de detección basado en física.
        """
        logger.info("========== INICIANDO PIPELINE ==========")
        
        _, df_not_scaled, _             = WaterPreprocessor.process_all_data()
        df_final, rf_model, rf_features = ModeloFisico.process(df_not_scaled, 
                                                               feature_names=list(FeatureConfig.PIPELINE_FEATURES.keys()))

        WaterApp._save_results(df_final, rf_model, rf_features)
        return df_final

   

    @staticmethod
    def _save_results(df_resultados: pd.DataFrame, rf_model: object, rf_features: list) -> None:
        logger.info(f"Guardando dataframe final consolidado en {Paths.PROC_CSV_AMAEM_NOT_SCALED}")
        df_resultados.to_csv(Paths.PROC_CSV_AMAEM_NOT_SCALED, index=False)


def main():
    parser = argparse.ArgumentParser(description="Water2Fraud Physical Pipeline")
    parser.add_argument("--run", action="store_true", help="Ejecutar el pipeline físico completo")
    args = parser.parse_args()

    if args.run:
        WaterApp.run_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()