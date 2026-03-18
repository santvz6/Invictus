import pandas as pd
import numpy as np
import argparse

import torch
import torch.nn as nn

from datetime import datetime

from src.water2fraud.features.preprocessor import WaterPreprocessor
from src.water2fraud.models.clustering import ClusterManager
from src.water2fraud.models.autoencoder import LSTMAutoencoder
from src.water2fraud.models.dataset import get_dataloader
from src.water2fraud.models.trainer import (
    train_autoencoder, 
    detect_anomalies, 
    plot_training_history, 
    plot_reconstruction
)

from src.config import get_logger, Paths, DatasetKeys, AIConstants
Paths.init_project()
logger = get_logger(__name__)


class WaterApp:
    """
    Orquestador principal del pipeline de Machine Learning 'Water2Fraud'.
    
    Esta clase gestiona el flujo de extremo a extremo (End-to-End) para la detección
    de viviendas turísticas ilegales en Alicante. Combina técnicas de procesamiento
    de series temporales, clustering dinámico (TimeSeriesKMeans) y arquitecturas 
    Deep Learning (LSTM-Autoencoders) junto con validaciones físicas.
    """

    @staticmethod
    def run_pipeline() -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de detección de fraude.
        
        Sigue las siguientes fases:
        0. Carga de datos base.
        1. Preprocesamiento y creación de secuencias temporales.
        2. Clustering de comportamientos de consumo.
        3. Entrenamiento de modelos LSTM-AE.
        4. Detección de anomalías.
        5. Persistencia de resultados y modelos entrenados.

        Returns:
            pd.DataFrame: DataFrame con los resultados completos de la detección, 
                          incluyendo puntuaciones de error y banderas de alerta.
        """
        logger.info("========== INICIANDO PIPELINE WATER2FRAUD (DEEP LEARNING) ==========")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Dispositivo de cómputo detectado: {device.upper()}")

        # FASE 0: Carga de datos
        df_raw = WaterApp._load_data()
        # FASE 1: Preprocesamiento y Secuenciación (Ventanas de 12 meses)
        X_sequences, metadata_df, feature_names = WaterApp._phase_1_preprocessing(df_raw)
        # FASE 2: Clustering Temporal de Series
        labels, cluster_manager = WaterApp._phase_2_clustering(X_sequences)
        metadata_df['cluster'] = labels
        # FASE 3: Entrenamiento de Autoencoders por Clúster
        modelos_entrenados = WaterApp._phase_3_training(X_sequences, metadata_df, device)
        # FASE 4: Detección de Anomalías e Inyección de Reglas Físicas
        df_resultados = WaterApp._phase_4_detection(X_sequences, metadata_df, modelos_entrenados, feature_names, device)
        # FASE 5: Guardado de resultados y modelos
        WaterApp._save_results(df_resultados, cluster_manager, modelos_entrenados)
        
        return df_resultados

    @staticmethod
    def _phase_1_preprocessing(df_raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
        """
        Limpia los datos crudos y genera las secuencias temporales.

        Args:
            df_raw (pd.DataFrame): Datos originales recién cargados.

        Returns:
            tuple:
                - X_sequences (np.ndarray): Tensor 3D para la red neuronal.
                - metadata_df (pd.DataFrame): Metadatos asociados a las secuencias.
                - feature_names (list[str]): Nombres de las variables en el orden de X_sequences
        """
        logger.info("--- FASE 1: Preprocesamiento y Secuencias Temporales ---")
        df_clean = WaterPreprocessor.process_raw_data(df_raw)

        # ! Aquí ya hemos cruzado df_clean con las fórmulas físicas y AEMET 
        # ! antes de crear las secuencias. WaterPreprocessor lo maneja.
        
        X_sequences, metadata_df, feature_names = WaterPreprocessor.create_sequences(df_clean, sequence_length=12)
        return X_sequences, metadata_df, feature_names

    @staticmethod
    def _phase_2_clustering(X_sequences: np.ndarray) -> tuple[np.ndarray, ClusterManager]:
        """
        Agrupa las secuencias en clústeres según la forma de su curva de consumo.

        Args:
            X_sequences (np.ndarray): Tensor 3D con las secuencias temporales.

        Returns:
            tuple:
                - labels (np.ndarray): Etiquetas de clúster asignadas.
                - cluster_manager (ClusterManager): Instancia del modelo de clustering entrenado.
        """
        logger.info("--- FASE 2: Clustering Temporal (TimeSeriesKMeans) ---")
        cluster_manager = ClusterManager(n_clusters=AIConstants.N_CLUSTERS_DEFAULT)
        labels = cluster_manager.fit_predict(X_sequences)
        logger.info(f"Distribución de clústeres: {pd.Series(labels).value_counts().to_dict()}")
        return labels, cluster_manager

    @staticmethod
    def _phase_3_training(X_sequences: np.ndarray, metadata_df: pd.DataFrame, 
                          device: str, **kwargs) -> dict[str, nn.Module]:
        """
        Fase de Entrenamiento: Construye y entrena un Autoencoder independiente para cada clúster.
        
        Aprende el comportamiento base (normal) de consumo de agua para los
        distintos tipos de vecindarios y perfiles aglutinados.

        Args:
            X_sequences (np.ndarray): Tensor 3D con todas las secuencias.
            metadata_df (pd.DataFrame): Metadatos que incluyen el clúster asignado.
            device (str): Dispositivo de cómputo ('cpu' o 'cuda').

        Returns:
            dict: Diccionario mapeando el ID del clúster con su respectivo modelo entrenado.
        """
        logger.info("--- FASE 3: Entrenamiento de Modelos LSTM-AE ---")
        
        # Kwargs
        batch_size  = kwargs.get("batch_size", 32)
        hidden_dim  = kwargs.get("hidden_dim", 64)
        latent_dim  = kwargs.get("latent_dim", 32) # 16
        epochs      = kwargs.get("epochs", 100)
        lr          = kwargs.get("lr", 1e-3)
        plot_graphs = kwargs.get("plot", False)
          

        modelos = {}
        clusters_unicos = metadata_df['cluster'].unique()
        num_features = X_sequences.shape[2]
        seq_len      = X_sequences.shape[1]

        for cluster_id in sorted(clusters_unicos):
            logger.info(f"> Entrenando Autoencoder para Clúster {cluster_id}...")
            
            # Filtrar datos de entrenamiento para este clúster
            idx_cluster = metadata_df['cluster'] == cluster_id
            X_cluster = X_sequences[idx_cluster]
            
            # Preparar DataLoader y Modelo
            dataloader = get_dataloader(X_cluster, batch_size=batch_size, shuffle=True)
            model = LSTMAutoencoder(num_features=num_features, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len)
            
            # Entrenar y almacenar
            model, history = train_autoencoder(model, dataloader, epochs=epochs, lr=lr, device=device)
            modelos[f"ae_cluster_{cluster_id}"] = model
            
            # Ploteamos si se solicitó en el Notebook
            if plot_graphs:
                plot_training_history(history, title=f"Evolución del Error - Clúster {cluster_id}")
  
        return modelos
    
    @staticmethod
    def _phase_4_detection(X_sequences: np.ndarray, metadata_df: pd.DataFrame, 
                           modelos: dict, feature_names: list[str], device: str) -> pd.DataFrame:
        """
        Fase de Inferencia y Detección: Evalúa los datos a través de los modelos entrenados
        y aplica las restricciones de las leyes físicas.

        Args:
            X_sequences (np.ndarray): Tensor 3D con todas las secuencias a evaluar.
            metadata_df (pd.DataFrame): Metadatos de los recibos de agua.
            modelos (dict): Diccionario con los modelos pre-entrenados por clúster.
            feature_names (list[str], optional): Nombres de las variables en el orden de X_sequences.
                                                 Si se omite, se usará 'feature_0', 'feature_1', etc.
            device (str): Dispositivo de cómputo ('cpu' o 'cuda').

        Returns:
            pd.DataFrame: Resultados completos de la detección (errores y alertas).
        """
        logger.info("--- FASE 4: Inferencia y Detección de Anomalías Turísticas ---")
        
        resultados_finales = []
        clusters_unicos = metadata_df['cluster'].unique()

        for cluster_id in sorted(clusters_unicos):
            logger.info(f"> Evaluando viviendas del Clúster {cluster_id}...")
            
            # Recuperar datos y modelo correspondiente
            idx_cluster = metadata_df['cluster'] == cluster_id
            X_cluster = X_sequences[idx_cluster]
            meta_cluster = metadata_df[idx_cluster].copy()
            
            model_key = f"ae_cluster_{cluster_id}"
            if model_key not in modelos:
                logger.warning(f"  Modelo para clúster {cluster_id} no encontrado. Se omitirá.")
                continue
                
            model = modelos[model_key]
            
            # Detección (Error de reconstrucción + Leyes Físicas)
            # Umbral físico: Si el consumo real supera 1.5x el teórico, es sospechoso
            df_anomalias = detect_anomalies(model, X_cluster, meta_cluster, 
                                            feature_names=feature_names, physics_threshold=1.5, device=device)
            resultados_finales.append(df_anomalias)
            
        # Unificar todos los resultados
        df_resultados = pd.concat(resultados_finales, ignore_index=True)
        return df_resultados

    @staticmethod
    def _load_data() -> pd.DataFrame:
        """
        Carga el dataset bruto utilizando las rutas definidas en la configuración del proyecto.

        Returns:
            pd.DataFrame: DataFrame con los datos brutos cargados.
            
        Raises:
            FileNotFoundError: Si el archivo CSV no se encuentra en la ruta especificada.
        """
        input_path = Paths.RAW_CSV_AMAEM
        
        if not input_path.exists():
            logger.error(f"Error crítico: No se encuentra el archivo en {input_path}")
            raise FileNotFoundError(f"Error crítico: No se encuentra el archivo en {input_path}")
        
        logger.info(f"Cargando datos desde {input_path}...")
        return pd.read_csv(input_path)

    @staticmethod
    def _save_results(df_resultados: pd.DataFrame, cluster_manager: ClusterManager, modelos: dict) -> None:
        """
        Persiste los resultados de la detección y los modelos entrenados.
        
        Crea una carpeta con un sello temporal (timestamp) donde guarda:
        - CSV con alertas confirmadas.
        - CSV con los resultados completos.
        - Archivo .joblib del modelo de clustering.
        - Archivos .pth con los pesos de los Autoencoders.

        Args:
            df_resultados (pd.DataFrame): DataFrame con la salida del pipeline de detección.
            cluster_manager (ClusterManager): Modelo K-Means entrenado.
            modelos (dict): Diccionario de modelos LSTMAutoencoder entrenados.
        """
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
    """
    Punto de entrada de la aplicación. Parsea los argumentos de la línea de comandos
    y lanza la ejecución correspondiente (análisis de codo o pipeline principal).
    """
    parser = argparse.ArgumentParser(description="Water2Fraud Deep Learning Orchestrator")
    parser.add_argument("--run", action="store_true", help="Ejecutar el pipeline completo")
    parser.add_argument("--elbow", action="store_true", help="Ejecutar análisis del codo para buscar el K óptimo")
    args = parser.parse_args()

    if args.elbow:
        df_raw = WaterApp._load_data()
        if df_raw is not None:
            X_sequences, _, _ = WaterApp._phase_1_preprocessing(df_raw)
            ClusterManager.find_optimal_clusters(X_sequences, max_clusters=10)
            
    elif args.run:
        WaterApp.run_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()