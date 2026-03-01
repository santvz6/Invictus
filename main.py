import pandas as pd
import argparse
from datetime import datetime

from src.water2fraud.features.preprocessor import WaterPreprocessor
from src.water2fraud.models.water_segmenter import WaterSegmenter
from src.water2fraud.cluster_trainer import WaterClusterTrainer
from src.water2fraud.models.labeler import WaterLabeler
from src.water2fraud.models.fraud_detector import FraudDetector
from src.config import get_logger, Paths, DatasetKeys, AIConstants

Paths.init_project()
logger = get_logger(__name__)

class WaterApp:
    @staticmethod
    def run_pipeline():
        logger.info("========== INICIANDO PIPELINE WATER2FRAUD ==========")
        
        # Preparación
        df_raw = WaterApp._load_data()
        if df_raw is None: return

        # Fases
        results, trainer = WaterApp._phase_1_segmentation(df_raw)
        results = WaterApp._phase_2_labeling(results, trainer)
        df_fraudes = WaterApp._phase_3_4_detection(results, df_raw)

        # Guardado
        WaterApp._save_results(df_fraudes, trainer)
        
        return df_fraudes

    @staticmethod
    def _phase_1_segmentation(df_raw):
        logger.info("--- FASE 1: Segmentación de Usuarios ---")
        preprocessor = WaterPreprocessor()
        segmenter = WaterSegmenter(n_clusters=AIConstants.N_CLUSTERS_DEFAULT)
        trainer = WaterClusterTrainer(segmenter, preprocessor)
        
        results = trainer.run_pipeline(df_raw)
        return results, trainer

    @staticmethod
    def _phase_2_labeling(results, trainer):
        logger.info("--- FASE 2: Etiquetado de Perfiles ---")
        centroides = trainer.model.get_centroids()
        feature_names: list[str] = DatasetKeys.get_feature_columns()
        
        cluster_labels_map: dict[str, str] = WaterLabeler.define_labels(centroides, feature_names)
        results[DatasetKeys.ETIQUETA_IA] = results[DatasetKeys.CLUSTER].astype(str).map(cluster_labels_map)
        return results

    @staticmethod
    def _phase_3_4_detection(results, df_raw):
        logger.info("--- FASE 3 y 4: Detección de Fraude ---")
        df_contratos = df_raw.groupby(DatasetKeys.ID_CONTADOR)[DatasetKeys.CONTRATO].first() # Agrupamos los mismos ID en un contrato
        results = results.join(df_contratos, on=DatasetKeys.ID_CONTADOR) # añadimos la columna del contrato para cada ID
        
        detector = FraudDetector(contamination=AIConstants.ISO_FOREST_CONTAMINATION)
        feature_names: list[str] = DatasetKeys.get_feature_columns()
        
        return detector.run_detection_pipeline(
            df_results=results,
            col_label_ia=DatasetKeys.ETIQUETA_IA,
            col_contrato=DatasetKeys.CONTRATO,
            feature_cols=feature_names
        )
    
    @staticmethod
    def _load_data():
        """Carga el dataset bruto y valida su existencia"""
        input_path = Paths.DATA_DIR / "AMAEM.csv"
        
        if not input_path.exists():
            logger.error(f"Error crítico: No se encuentra el archivo en {input_path}")
            return None
        
        logger.info(f"Cargando datos desde {input_path}...")
        return pd.read_csv(input_path)

    @staticmethod
    def _save_results(df_fraudes, trainer: WaterClusterTrainer):
        """Gestiona la persistencia de resultados y artefactos visuales"""

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = Paths.EXPERIMENTS_DIR / timestamp
        folder_path.mkdir(parents=True, exist_ok=True)
        
        csv_name = "deteccion_fraude_final.csv"
        df_fraudes.to_csv(folder_path / csv_name, index=True)

        trainer.save_artifacts(output_path=folder_path)

        print(f"\n{'='*50}")
        print(f"PROCESO FINALIZADO CON ÉXITO")
        print(f"Carpeta de salida: {folder_path}")
        print(f"Casos sospechosos detectados: {len(df_fraudes)}")
        print(f"{'='*50}")
        
        print("\nTOP 5 SOSPECHOSOS (PRIORIDAD ALTA):")
        cols_to_show = [DatasetKeys.STATUS, DatasetKeys.CONFIDENCE]
        print(df_fraudes[cols_to_show].head(5))

def main():
    parser = argparse.ArgumentParser(description="Water2Fraud Pipeline Orchestrator")
    parser.add_argument("--run", action="store_true", help="Ejecutar el pipeline completo")
    args = parser.parse_args()

    if args.run:
        WaterApp.run_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()