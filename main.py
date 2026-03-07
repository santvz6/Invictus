import pandas as pd
import argparse
import torch
from datetime import datetime

# Importamos las nuevas clases que diseñamos
from src.water2fraud.features.preprocessor import WaterPreprocessor
from src.water2fraud.models.clustering import ClusterManager
from src.water2fraud.models.autoencoder import LSTMAutoencoder
from src.water2fraud.models.trainer import train_autoencoder, detect_anomalies
from src.water2fraud.models.dataset import get_dataloader
from src.config import get_logger, Paths, DatasetKeys, AIConstants

# Inicializamos directorios
Paths.init_project()
logger = get_logger(__name__)

class WaterApp:
    @staticmethod
    def run_pipeline():
        logger.info("========== INICIANDO PIPELINE WATER2FRAUD (DEEP LEARNING) ==========")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Dispositivo de cómputo detectado: {device.upper()}")

        # FASE 0: Carga de datos
        df_raw = WaterApp._load_data()
        if df_raw is None: return

        # FASE 1: Preprocesamiento y Secuenciación (Ventanas de 12 meses)
        X_sequences, metadata_df = WaterApp._phase_1_preprocessing(df_raw)

        # FASE 2: Clustering Temporal de Series
        labels, cluster_manager = WaterApp._phase_2_clustering(X_sequences)
        metadata_df['cluster'] = labels

        # FASE 3 y 4: Entrenamiento de Autoencoders y Detección
        df_resultados, modelos_entrenados = WaterApp._phase_3_4_train_and_detect(
            X_sequences, metadata_df, device
        )

        # FASE 5: Guardado de resultados y modelos
        WaterApp._save_results(df_resultados, cluster_manager, modelos_entrenados)
        
        return df_resultados

    @staticmethod
    def _phase_1_preprocessing(df_raw):
        logger.info("--- FASE 1: Preprocesamiento y Secuencias Temporales ---")
        df_clean = WaterPreprocessor.process_raw_data(df_raw)
        
        # OJO: Aquí deberías haber cruzado df_clean con las fórmulas físicas y AEMET 
        # antes de crear las secuencias. Asumimos que WaterPreprocessor lo maneja.
        
        X_sequences, metadata_df = WaterPreprocessor.create_sequences(df_clean, sequence_length=12)
        return X_sequences, metadata_df

    @staticmethod
    def _phase_2_clustering(X_sequences):
        logger.info("--- FASE 2: Clustering Temporal (TimeSeriesKMeans) ---")
        cluster_manager = ClusterManager(n_clusters=AIConstants.N_CLUSTERS_DEFAULT)
        labels = cluster_manager.fit_predict(X_sequences)
        
        logger.info(f"Distribución de clústeres: {pd.Series(labels).value_counts().to_dict()}")
        return labels, cluster_manager

    @staticmethod
    def _phase_3_4_train_and_detect(X_sequences, metadata_df, device):
        logger.info("--- FASE 3 y 4: Entrenamiento LSTM-AE y Detección de Anomalías ---")
        
        resultados_finales = []
        modelos = {}
        clusters_unicos = metadata_df['cluster'].unique()
        
        # Obtenemos el número de features de nuestra matriz X
        num_features = X_sequences.shape[2]

        for cluster_id in sorted(clusters_unicos):
            logger.info(f"> Procesando Clúster {cluster_id}...")
            
            # 1. Filtrar datos del clúster actual
            idx_cluster = metadata_df['cluster'] == cluster_id
            X_cluster = X_sequences[idx_cluster]
            meta_cluster = metadata_df[idx_cluster].copy()
            
            # 2. DataLoader
            dataloader = get_dataloader(X_cluster, batch_size=32, shuffle=True)
            
            # 3. Inicializar y Entrenar Autoencoder
            model = LSTMAutoencoder(num_features=num_features, hidden_dim=64, latent_dim=16, seq_len=12)
            logger.info(f"  Entrenando modelo para clúster {cluster_id}...")
            model = train_autoencoder(model, dataloader, epochs=50, lr=1e-3, device=device)
            modelos[f"ae_cluster_{cluster_id}"] = model
            
            # 4. Detección (Error de reconstrucción + Leyes Físicas)
            logger.info(f"  Evaluando anomalías en clúster {cluster_id}...")
            # Umbral físico: Si el consumo real supera 1.5x el teórico, es sospechoso
            df_anomalias = detect_anomalies(model, X_cluster, meta_cluster, physics_threshold=1.5, device=device)
            resultados_finales.append(df_anomalias)
            
        # Unimos los resultados de todos los clústeres
        df_resultados = pd.concat(resultados_finales, ignore_index=True)
        return df_resultados, modelos

    @staticmethod
    def _load_data():
        """Carga el dataset bruto utilizando las rutas definidas en Paths"""
        input_path = Paths.RAW_CSV_AMAEM
        
        if not input_path.exists():
            logger.error(f"Error crítico: No se encuentra el archivo en {input_path}")
            return None
        
        logger.info(f"Cargando datos desde {input_path}...")
        return pd.read_csv(input_path)

    @staticmethod
    def _save_results(df_resultados, cluster_manager, modelos):
        """Guarda el CSV de fraudes y los modelos entrenados"""
        logger.info("--- FASE 5: Guardando Resultados ---")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = Paths.EXPERIMENTS_DIR / timestamp
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Filtrar y guardar solo las anomalías confirmadas
        if 'ALERTA_TURISTICA_ILEGAL' in df_resultados.columns:
            df_fraudes = df_resultados[df_resultados['ALERTA_TURISTICA_ILEGAL'] == True]
        else:
            df_fraudes = df_resultados[df_resultados['is_ae_anomaly'] == True] # Fallback si no hay física
            
        csv_name = "alertas_fraude_turistico.csv"
        df_fraudes.to_csv(folder_path / csv_name, index=False)
        
        # 2. Guardar dataset completo con scores para análisis
        df_resultados.to_csv(folder_path / "resultados_completos.csv", index=False)

        # 3. Guardar Modelos (Clustering y AEs)
        cluster_manager.save(folder_path / "ts_kmeans_model.joblib")
        for name, model in modelos.items():
            torch.save(model.state_dict(), folder_path / f"{name}.pth")

        print(f"\n{'='*60}")
        print(f"PROCESO DEEP LEARNING FINALIZADO CON ÉXITO")
        print(f"Carpeta de salida: {folder_path}")
        print(f"Viviendas analizadas: {len(df_resultados)}")
        print(f"Casos SOSPECHOSOS DETECTADOS: {len(df_fraudes)}")
        print(f"{'='*60}")
        
        if not df_fraudes.empty:
            print("\nTOP 5 ALERTA ROJA (Alto Error de Reconstrucción + Violación Física):")
            # Ajusta las columnas según lo que tengas en metadata_df
            cols = [DatasetKeys.BARRIO, DatasetKeys.FECHA, 'reconstruction_error']
            cols = [c for c in cols if c in df_fraudes.columns]
            print(df_fraudes.sort_values(by='reconstruction_error', ascending=False)[cols].head(5))

def main():
    parser = argparse.ArgumentParser(description="Water2Fraud Deep Learning Orchestrator")
    parser.add_argument("--run", action="store_true", help="Ejecutar el pipeline completo")
    args = parser.parse_args()

    if args.run:
        WaterApp.run_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()