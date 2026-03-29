import pandas as pd
import argparse
import torch

from pathlib import Path
from datetime import datetime

from src.features.preprocessor import WaterPreprocessor
from src.features.fisicos_processor import FisicosProcessor
from src.config import get_logger, Paths, DatasetKeys

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        df_scaled, df_not_scaled, scalers = WaterPreprocessor.process_all_data()
        df_final, rf_model, rf_features = FisicosProcessor.process(df_not_scaled, feature_names=list(WaterPreprocessor.FEATURES.keys()))

        WaterApp._save_results(df_final, rf_model, rf_features)
        return df_final

   

    @staticmethod
    def _save_results(df_resultados: pd.DataFrame, rf_model: object, rf_features: list) -> None:
        pass


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