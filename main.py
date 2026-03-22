import pandas as pd
import numpy as np
import argparse

import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime

from src.water2fraud.features.preprocessor import WaterPreprocessor
from src.water2fraud.models.clustering import ClusterManager
from src.water2fraud.models.autoencoder import LSTMAutoencoder
from src.water2fraud.models.dataset import get_dataloader
from src.water2fraud.models.trainer import (
    train_autoencoder, 
    detect_ae_anomalies, 
    plot_training_history, 
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
        metadata_df[DatasetKeys.CLUSTER] = labels
        # FASE 3: Entrenamiento de Autoencoders por Clúster
        modelos_entrenados = WaterApp._phase_3_training(X_sequences, metadata_df, device)
        # FASE 4: Inferencia y Detección de Anomalías (Autoencoder)
        df_ae_resultados = WaterApp._phase_4_ae_detection(X_sequences, metadata_df, modelos_entrenados, feature_names, device)
        # FASE 5: Ponderación Física y Cálculo de Riesgo de Fraude (Risk Score)
        df_final = WaterApp._phase_5_risk_scoring(df_ae_resultados)
        # FASE 6: Guardado de resultados y modelos
        WaterApp._save_results(df_final, cluster_manager, modelos_entrenados)
        
        return df_final

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
        df_scaled = WaterPreprocessor.process_raw_data(df_raw)

        # ! Aquí ya hemos cruzado df_scaled con los features.
        # ! Antes de crear las secuencias, WaterPreprocessor lo maneja.
        
        X_sequences, metadata_df, feature_names = WaterPreprocessor.create_sequences(df_scaled, sequence_length=12)
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
        batch_size  = kwargs.get("batch_size", 64)
        hidden_dim  = kwargs.get("hidden_dim", 128)
        latent_dim  = kwargs.get("latent_dim", 16)
        epochs      = kwargs.get("epochs", 150)
        lr          = kwargs.get("lr", 1e-3)
        plot_graphs = kwargs.get("plot", False)
          

        modelos = {}
        clusters_unicos = metadata_df[DatasetKeys.CLUSTER].unique()
        num_features = X_sequences.shape[2]
        seq_len      = X_sequences.shape[1]

        for cluster_id in sorted(clusters_unicos):
            logger.info(f"> Entrenando Autoencoder para Clúster {cluster_id}...")
            
            # Filtrar datos de entrenamiento para este clúster
            idx_cluster = metadata_df[DatasetKeys.CLUSTER] == cluster_id
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
    def _phase_4_ae_detection(X_sequences: np.ndarray, metadata_df: pd.DataFrame, 
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
        clusters_unicos = metadata_df[DatasetKeys.CLUSTER].unique()

        for cluster_id in sorted(clusters_unicos):
            logger.info(f"> Evaluando viviendas del Clúster {cluster_id}...")
            
            # Recuperar datos y modelo correspondiente
            idx_cluster = metadata_df[DatasetKeys.CLUSTER] == cluster_id
            X_cluster = X_sequences[idx_cluster]
            meta_cluster = metadata_df[idx_cluster].copy()
            
            model_key = f"ae_cluster_{cluster_id}"
            if model_key not in modelos:
                logger.warning(f"  Modelo para clúster {cluster_id} no encontrado. Se omitirá.")
                continue
                
            model = modelos[model_key]
            
            # Detección Autoencoder
            df_anomalias = detect_ae_anomalies(model, X_cluster, meta_cluster, AIConstants.AE_ANOMALIES_PERCENTILE, 
                                               feature_names=feature_names, device=device)
            resultados_finales.append(df_anomalias) 
            
        # Unificar todos los resultados
        df_ae_resultados = pd.concat(resultados_finales, ignore_index=True)
        return df_ae_resultados

    @staticmethod
    def _phase_5_risk_scoring(df_ae: pd.DataFrame, risk_score={"ae_weight": 0.8, "physics_weight": 0.2}) -> pd.DataFrame:
        """
        Cruza los errores de reconstrucción del Autoencoder (AE) con las predicciones
        físicas sobre los datos reales (no escalados) para generar una puntuación
        de riesgo de fraude combinada (Ensamble manual).
        """
        logger.info("--- FASE 5: Ponderación Híbrida (AE + Física) y Puntuación de Riesgo ---")
        from src.water2fraud.features.fisicos_processor import FisicosProcessor
        from sklearn.preprocessing import MinMaxScaler
        
        # 1. Procesamos la lógica física sobre los datos NO ESCALADOS
        df_not_scaled = pd.read_csv(Paths.PROC_CSV_AMAEM_NOT_SCALED)
        df_fisicos    = FisicosProcessor.process(df_not_scaled, list(WaterPreprocessor.FEATURES.keys()))

        # 2. Cruzamos AE con Física
        cols_ae = [DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA, 
                    DatasetKeys.RECONSTRUCTION_ERROR, DatasetKeys.CLUSTER, DatasetKeys.IS_AE_ANOMALY]
        ae_consumo_col = f'signed_error__{DatasetKeys.CONSUMO_RATIO}'
        cols_ae.append(ae_consumo_col)

        df_final = pd.merge(
            df_fisicos, 
            df_ae[cols_ae], 
            on=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA], 
            how='inner'
        )

        # 3. Normalizamos Errores (0 a 100)
        scaler = MinMaxScaler(feature_range=(0, 100))
        df_final[DatasetKeys.AE_SCORE] = scaler.fit_transform(df_final[[DatasetKeys.RECONSTRUCTION_ERROR]])
        df_final.loc[df_final[ae_consumo_col] <= 0, DatasetKeys.AE_SCORE] = 0
        
        df_final[DatasetKeys.RESIDUO_POSITIVO] = df_final[DatasetKeys.RESIDUO].clip(lower=0)
        df_final[DatasetKeys.PHYSICS_SCORE] = scaler.fit_transform(df_final[[DatasetKeys.RESIDUO_POSITIVO]])

        # 4. Calculamos el FRAUD RISK SCORE
        df_final[DatasetKeys.FRAUD_RISK_SCORE] = (
             (df_final[DatasetKeys.AE_SCORE] * risk_score["ae_weight"]) + 
             (df_final[DatasetKeys.PHYSICS_SCORE] * risk_score["physics_weight"])
        )

        # 5. Definimos la Alerta
        umbral_rojo = np.percentile(df_final[DatasetKeys.FRAUD_RISK_SCORE], AIConstants.PHYSICS_ANOMALIES_PERCENTILE)
        df_final[DatasetKeys.ALERTA_TURISTICA_ILEGAL] = df_final[DatasetKeys.FRAUD_RISK_SCORE] > umbral_rojo

        # 6. Clasificamos Nivel de Riesgo
        df_final[DatasetKeys.NIVEL_RIESGO] = "Bajo"
        df_final.loc[df_final[DatasetKeys.FRAUD_RISK_SCORE] > 30, DatasetKeys.NIVEL_RIESGO] = "Medio"
        df_final.loc[df_final[DatasetKeys.FRAUD_RISK_SCORE] > 70, DatasetKeys.NIVEL_RIESGO] = "Alto"

        # 7. Generamos motivos explicativos
        df_final[DatasetKeys.MOTIVO] = WaterApp._generate_alert_reasons(df_final)

        # Limpiamos columnas auxiliares
        df_final = df_final.drop(columns=[DatasetKeys.RESIDUO_POSITIVO])
        
        return df_final

    @staticmethod
    def _generate_alert_reasons(df: pd.DataFrame) -> pd.Series:
        def format_reason(row):
            reasons = []
            ae_score = row.get(DatasetKeys.AE_SCORE, 0)
            phys_score = row.get(DatasetKeys.PHYSICS_SCORE, 0)
            
            if ae_score > 60: reasons.append(f"Pattern IA ({ae_score:.0f}%)")
            if phys_score > 30: reasons.append(f"Exceso Físico ({phys_score:.0f}%)")
            
            if not reasons and row[DatasetKeys.FRAUD_RISK_SCORE] > 0:
                return "Riesgo moderado acumulado."
            return " + ".join(reasons) if reasons else "Sin anomalías notables."
            
        return df.apply(format_reason, axis=1)

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
        """
        logger.info("--- FASE 5: Guardando Resultados y Generando Reportes ---")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = Paths.EXPERIMENTS_DIR / timestamp
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Alertas Priorizadas (Con nombres legibles y diferenciados)
        cols_final_renamed = {
            DatasetKeys.BARRIO: "Barrio",
            DatasetKeys.USO: "Uso",
            DatasetKeys.FECHA: "Fecha",
            DatasetKeys.CONSUMO_RATIO: "Consumo_Real_L_dia",
            DatasetKeys.RESIDUO: "Diferencia_Fisica_L",
            DatasetKeys.IS_AE_ANOMALY: "Alerta_Solo_Autoencoder",
            DatasetKeys.AE_SCORE: "Riesgo_Autoencoder_%",
            DatasetKeys.PHYSICS_SCORE: "Riesgo_Fisico_%",
            DatasetKeys.FRAUD_RISK_SCORE: "Puntaje_Hibrido_Final",
            DatasetKeys.NIVEL_RIESGO: "Nivel_Riesgo",
            DatasetKeys.MOTIVO: "Motivo_Explicacion"
        }
        
        cols_presentes = [c for c in cols_final_renamed.keys() if c in df_resultados.columns]
        df_alertas = df_resultados[df_resultados[DatasetKeys.ALERTA_TURISTICA_ILEGAL] == True].copy()
        df_alertas = df_alertas.sort_values(by=DatasetKeys.FRAUD_RISK_SCORE, ascending=False)
        
        # Guardar CSV Priorizado con nombres claros
        df_priorizadas = df_alertas[cols_presentes].rename(columns=cols_final_renamed)
        df_priorizadas.to_csv(folder_path / "alertas_priorizadas.csv", index=False)
        
        # 2. Guardar dataset completo (para científicos de datos)
        df_resultados.to_csv(folder_path / "resultados_completos_tecnicos.csv", index=False)

        # 3. Guardar Modelos
        cluster_manager.save(folder_path / "ts_kmeans_model.joblib")
        for name, model in modelos.items():
            torch.save(model.state_dict(), folder_path / f"{name}.pth")

        # 4. Generar Reporte Markdown Profesional
        WaterApp._generate_markdown_report(df_alertas, folder_path)

        print(f"\n{'='*60}")
        print(f"PIPELINE WATER2FRAUD FINALIZADO")
        print(f"Reporte de Análisis: {folder_path / 'REPORTE_ANALISIS.md'}")
        print(f"Casos críticos detectados: {len(df_alertas)}")
        print(f"{'='*60}")
        
        if not df_alertas.empty:
            print("\n🚨 TOP 5 CASOS DE ALTO RIESGO:")
            print(df_priorizadas.head(5).to_string(index=False))

    @staticmethod
    def _generate_markdown_report(df_alertas: pd.DataFrame, folder_path: Path) -> None:
        report_path = folder_path / "REPORTE_ANALISIS.md"
        from datetime import datetime
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 🏛️ REPORTE EJECUTIVO: Auditoría de Viviendas Turísticas\n\n")
            f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Alertas totales:** {len(df_alertas)}\n\n")
            f.write("## 📌 Top 10 Alertas de Fraude\n\n")
            f.write("| Barrio | Uso | Fecha | Riesgo (%) | Nivel | Motivo |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for _, row in df_alertas.head(10).iterrows():
                f.write(f"| {row[DatasetKeys.BARRIO]} | {row[DatasetKeys.USO]} | {row[DatasetKeys.FECHA]} | ")
                f.write(f"{row[DatasetKeys.FRAUD_RISK_SCORE]:.1f}% | {row[DatasetKeys.NIVEL_RIESGO]} | {row[DatasetKeys.MOTIVO]} |\n")
            f.write("\n\n---\n*Generado por Invictus Analytics Engine*")


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