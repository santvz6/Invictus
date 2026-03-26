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

from sklearn.model_selection import train_test_split
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
        X_sequences, metadata_df, feature_names, scalers = WaterApp._phase_1_preprocessing(df_raw)
        # FASE 2: Clustering Temporal de Series
        labels, cluster_manager = WaterApp._phase_2_clustering(X_sequences)
        metadata_df[DatasetKeys.CLUSTER] = labels
        # FASE 3: Entrenamiento de Autoencoders por Clúster
        modelos_entrenados = WaterApp._phase_3_training(X_sequences, metadata_df, device)
        # FASE 4: Inferencia y Detección de Anomalías (Autoencoder)
        df_ae_resultados, umbrales = WaterApp._phase_4_ae_detection(X_sequences, metadata_df, modelos_entrenados, feature_names, device)
        # FASE 5: Ponderación Física y Cálculo de Riesgo de Fraude (Risk Score)
        df_final, rf_model, rf_features = WaterApp._phase_5_risk_scoring(df_ae_resultados)
        # FASE 6: Guardado de resultados y modelos
        WaterApp._save_results(df_final, cluster_manager, modelos_entrenados, scalers, rf_model, rf_features, umbrales)
        
        return df_final

    @staticmethod
    def _phase_1_preprocessing(df_raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str], dict]:
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
        df_scaled, scalers = WaterPreprocessor.process_raw_data(df_raw)

        # ! Aquí ya hemos cruzado df_scaled con los features.
        # ! Antes de crear las secuencias, WaterPreprocessor lo maneja.
        
        X_sequences, metadata_df, feature_names = WaterPreprocessor.create_sequences(df_scaled, sequence_length=12)
        return X_sequences, metadata_df, feature_names, scalers

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
        batch_size   = kwargs.get("batch_size", 32)
        hidden_dim   = kwargs.get("hidden_dim", 128)
        latent_dim   = kwargs.get("latent_dim", 32)
        epochs       = kwargs.get("epochs", 170)
        lr           = kwargs.get("lr", 5e-4)
        weight_decay = kwargs.get("weight_decay", 1e-4) 
        plot_graphs  = kwargs.get("plot", False)
          

        modelos = {}
        clusters_unicos = metadata_df[DatasetKeys.CLUSTER].unique()
        num_features = X_sequences.shape[2]
        seq_len      = X_sequences.shape[1]

        for cluster_id in sorted(clusters_unicos):
            logger.info(f"> Entrenando Autoencoder para Clúster {cluster_id}...")
            
            # Filtrar datos de entrenamiento para este clúster
            idx_cluster = metadata_df[DatasetKeys.CLUSTER] == cluster_id
            X_cluster = X_sequences[idx_cluster]
            
            # Split de entrenamiento y validación (20% para validación)
            X_train, X_val = train_test_split(
                X_cluster, 
                test_size=0.2, 
                random_state=AIConstants.RANDOM_STATE
            )
            
            # Preparar DataLoaders y Modelo
            train_loader = get_dataloader(X_train, batch_size=batch_size, shuffle=True)
            val_loader = get_dataloader(X_val, batch_size=batch_size, shuffle=False)
            
            model = LSTMAutoencoder(num_features=num_features, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len)
            
            # Entrenar y almacenar (Pasamos ambos loaders)
            model, history = train_autoencoder(
                model, 
                train_loader, 
                val_loader=val_loader, 
                epochs=epochs, 
                lr=lr, 
                weight_decay=weight_decay,
                device=device
            )
            modelos[f"ae_cluster_{cluster_id}"] = model
            
            # Ploteamos si se solicitó en el Notebook
            if plot_graphs:
                plot_training_history(history, title=f"Evolución del Error - Clúster {cluster_id}")
  
        return modelos
    
    @staticmethod
    def _phase_4_ae_detection(X_sequences: np.ndarray, metadata_df: pd.DataFrame, 
                           modelos: dict, feature_names: list[str], device: str) -> tuple[pd.DataFrame, dict]:
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
        umbrales = {}
        clusters_unicos = metadata_df[DatasetKeys.CLUSTER].unique()

        # Ponderación para el error de reconstrucción (haciéndolo escalable)
        feature_weights = {
            DatasetKeys.CONSUMO_RATIO: 0.70,
            DatasetKeys.PCT_VT_SIN_REGISTRAR: 0.30
        }

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
            df_anomalias, umbral = detect_ae_anomalies(model, X_cluster, meta_cluster, 
                                               feature_names=feature_names, device=device,
                                               feature_weights=feature_weights)
            
            resultados_finales.append(df_anomalias) 
            umbrales[str(cluster_id)] = umbral
            
        # Unificar todos los resultados
        df_ae_resultados = pd.concat(resultados_finales, ignore_index=True)
        return df_ae_resultados, umbrales

    @staticmethod
    def _phase_5_risk_scoring(df_ae: pd.DataFrame, risk_score={"ae_weight": 0.2, "physics_weight": 0.8}) -> tuple[pd.DataFrame, object, list[str]]:
        """
        Cruza los errores de reconstrucción del Autoencoder (AE) con las predicciones
        físicas sobre los datos reales (no escalados) para generar una puntuación
        de riesgo de fraude combinada (Ensamble manual).
        """
        logger.info("--- FASE 5: Ponderación Híbrida (AE + Física) y Puntuación de Riesgo ---")
        from src.water2fraud.features.fisicos_processor import FisicosProcessor
        from sklearn.preprocessing import MinMaxScaler
        
        # === FILTRO ESTRATÉGICO ===
        logger.info("Filtrando análisis de fraude unicamente a contratos de uso DOMÉSTICO...")
        df_ae = df_ae[df_ae[DatasetKeys.USO] == "DOMESTICO"].copy()

        # 1. Procesamos la lógica física sobre los datos NO ESCALADOS
        df_not_scaled = pd.read_csv(Paths.PROC_CSV_AMAEM_NOT_SCALED)
        df_not_scaled = df_not_scaled[df_not_scaled[DatasetKeys.USO] == "DOMESTICO"].copy()
        
        df_fisicos, rf_model, rf_features = FisicosProcessor.process(df_not_scaled, list(WaterPreprocessor.FEATURES.keys()))

        # 2. Cruzamos AE con Física
        cols_ae = [DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA, 
                    DatasetKeys.CLUSTER,
                    DatasetKeys.IS_GENERAL_ANOMALY, DatasetKeys.IS_WEIGHTED_ANOMALY,
                    DatasetKeys.AE_SCORE_GENERAL, DatasetKeys.AE_SCORE_WEIGHTED]
        
        # Propagamos dinámicamente el error global y los desgloses por feature al dataset final
        cols_ae.extend([c for c in df_ae.columns if c.startswith("error__") or c == DatasetKeys.RECONSTRUCTION_ERROR])

        df_final = pd.merge(
            df_fisicos, 
            df_ae[cols_ae], 
            on=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA], 
            how='inner'
        )

        # 3. Normalizamos Errores Físicos (0 a 100)
        scaler = MinMaxScaler(feature_range=(0, 100))
        
        # El AE_SCORE_WEIGHTED ya viene normalizado localmente (100 = umbral) desde Phase 4
        
        df_final[DatasetKeys.RESIDUO_POSITIVO] = df_final[DatasetKeys.RESIDUO].clip(lower=0)
        df_final[DatasetKeys.PHYSICS_SCORE] = scaler.fit_transform(df_final[[DatasetKeys.RESIDUO_POSITIVO]])

        umbral_fisico = np.percentile(df_final[DatasetKeys.PHYSICS_SCORE], AIConstants.FRAUD_RISK_PERCENTILE)
        df_final[DatasetKeys.IS_PHYSICS_ANOMALY] = df_final[DatasetKeys.PHYSICS_SCORE] > umbral_fisico

        # 4. Calculamos el FRAUD RISK SCORE
        df_final[DatasetKeys.FRAUD_RISK_SCORE] = (
             (df_final[DatasetKeys.AE_SCORE_WEIGHTED] * risk_score["ae_weight"]) + 
             (df_final[DatasetKeys.PHYSICS_SCORE] * risk_score["physics_weight"])
        )

        # 5. Definimos la Alerta
        umbral_rojo = np.percentile(df_final[DatasetKeys.FRAUD_RISK_SCORE], AIConstants.FRAUD_RISK_PERCENTILE)
        df_final[DatasetKeys.ALERTA_TURISTICA_ILEGAL] = df_final[DatasetKeys.FRAUD_RISK_SCORE] > umbral_rojo

        # 6. Clasificamos Nivel de Riesgo
        df_final[DatasetKeys.NIVEL_RIESGO] = "Bajo"
        df_final.loc[df_final[DatasetKeys.FRAUD_RISK_SCORE] > 30, DatasetKeys.NIVEL_RIESGO] = "Medio"
        df_final.loc[df_final[DatasetKeys.FRAUD_RISK_SCORE] > 70, DatasetKeys.NIVEL_RIESGO] = "Alto"

        # 7. Generamos motivos explicativos
        df_final[DatasetKeys.MOTIVO] = WaterApp._generate_alert_reasons(df_final)

        # Limpiamos columnas auxiliares
        df_final = df_final.drop(columns=[DatasetKeys.RESIDUO_POSITIVO])
        
        return df_final, rf_model, rf_features

    @staticmethod
    def _generate_alert_reasons(df: pd.DataFrame) -> pd.Series:
        def format_reason(row):
            reasons = []
            ae_score = row.get(DatasetKeys.AE_SCORE_WEIGHTED, 0)
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
    def _save_results(df_resultados: pd.DataFrame, cluster_manager: ClusterManager, modelos: dict, 
                      scalers: dict, rf_model: object, rf_features: list, umbrales: dict) -> None:
        """
        Persiste los resultados de la detección y los modelos entrenados.
        """
        logger.info("--- FASE 5: Guardando Resultados y Generando Reportes ---")
        
        import joblib
        import json
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = Paths.EXPERIMENTS_DIR / timestamp
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Estrategia de partición de CSVs
        base_cols = [DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA]
        
        # 1. Alertas Priorizadas (Solo fraude detectado de alto impacto)
        df_alertas = df_resultados[df_resultados[DatasetKeys.ALERTA_TURISTICA_ILEGAL] == True].copy()
        df_alertas = df_alertas.sort_values(by=DatasetKeys.FRAUD_RISK_SCORE, ascending=False)    
        df_alertas.to_csv(folder_path / "01_alertas_priorizadas.csv", index=False)
        
        # 2. Resultados Generales (Resumen Ejecutivo)
        general_cols = base_cols + [DatasetKeys.FRAUD_RISK_SCORE, DatasetKeys.NIVEL_RIESGO, DatasetKeys.MOTIVO, DatasetKeys.ALERTA_TURISTICA_ILEGAL]
        df_generales = df_resultados[[c for c in general_cols if c in df_resultados.columns]]
        df_generales.to_csv(folder_path / "02_resultados_generales.csv", index=False)
        
        # 3. Métricas Físicas (Predicciones y Residuos)
        fisica_cols = base_cols + [DatasetKeys.CONSUMO_FISICO_ESPERADO, DatasetKeys.PREDICCION_FOURIER, DatasetKeys.IMPACTO_EXOGENO, DatasetKeys.RESIDUO, DatasetKeys.PHYSICS_SCORE, DatasetKeys.IS_PHYSICS_ANOMALY]
        df_fisica = df_resultados[[c for c in fisica_cols if c in df_resultados.columns]]
        df_fisica.to_csv(folder_path / "03_metricas_fisicas.csv", index=False)
        
        # 4. Métricas Autoencoder (IA y Anomalías Principales)
        ae_cols = base_cols + [DatasetKeys.CLUSTER, DatasetKeys.AE_SCORE_GENERAL, DatasetKeys.AE_SCORE_WEIGHTED, DatasetKeys.IS_GENERAL_ANOMALY, DatasetKeys.IS_WEIGHTED_ANOMALY]
        df_ae = df_resultados[[c for c in ae_cols if c in df_resultados.columns]]
        df_ae.to_csv(folder_path / "04_metricas_autoencoder.csv", index=False)
        
        # 5. Desglose de Errores de Reconstrucción (por Feature)
        ae_extra = [c for c in df_resultados.columns if c.startswith("error__") and c not in ae_cols]
        feat_err_cols = base_cols + [DatasetKeys.CLUSTER] + ae_extra
        df_feat_errors = df_resultados[[c for c in feat_err_cols if c in df_resultados.columns]]
        df_feat_errors.to_csv(folder_path / "05_errores_reconstruccion_features.csv", index=False)
        
        # 6. Dataset Completo (Para Científicos de Datos)
        df_resultados.to_csv(folder_path / "06_resultados_completos_tecnicos.csv", index=False)

        # 7. Resumen de Scores (Solo Barrio, Fecha y Puntuaciones clave)
        score_cols = [DatasetKeys.BARRIO, DatasetKeys.FECHA, DatasetKeys.AE_SCORE_WEIGHTED, DatasetKeys.PHYSICS_SCORE, DatasetKeys.FRAUD_RISK_SCORE]
        df_scores = df_resultados[[c for c in score_cols if c in df_resultados.columns]]
        df_scores.to_csv(folder_path / "07_resumen_scores.csv", index=False)

        # 8. Guardar Modelos y Artefactos (Caché Dashboard)
        cluster_manager.save(folder_path / "ts_kmeans_model.joblib")
        for name, model in modelos.items():
            checkpoint = {
                "state_dict": model.state_dict(),
                "num_features": model.num_features,
                "hidden_dim": model.hidden_dim,
                "latent_dim": model.latent_dim,
                "seq_len": model.seq_len
            }
            torch.save(checkpoint, folder_path / f"{name}.pth")

        joblib.dump(scalers, folder_path / "scalers.joblib")
        joblib.dump(rf_model, folder_path / "rf_model.joblib")
        
        with open(folder_path / "rf_features.json", "w") as f:
            json.dump(rf_features, f)
            
        with open(folder_path / "thresholds.json", "w") as f:
            json.dump(umbrales, f)


        # 9. Generar Reporte Markdown Profesional
        WaterApp._generate_markdown_report(df_alertas, folder_path)

        print(f"\n{'='*60}")
        print(f"PIPELINE WATER2FRAUD FINALIZADO")
        print(f"Reporte de Análisis: {folder_path / 'REPORTE_ANALISIS.md'}")
        print(f"Casos críticos detectados: {len(df_alertas)}")
        print(f"{'='*60}")
        
        if not df_alertas.empty:
            print("\n🚨 TOP 5 CASOS DE ALTO RIESGO:")
            print(df_alertas.head(5).to_string(index=False))

    @staticmethod
    def _generate_markdown_report(df_alertas: pd.DataFrame, folder_path: Path) -> None:
        report_path = folder_path / "REPORTE_ANALISIS.md"
        
        # Filtramos barrios dispersos del reporte visual pero se mantienen en los datos técnicos
        df_reporte = df_alertas[~df_alertas[DatasetKeys.BARRIO].str.contains("DISPERSOS", case=False, na=False)].copy()
        
        from datetime import datetime
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 🏛️ REPORTE EJECUTIVO: Auditoría de Viviendas Turísticas\n\n")
            f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Alertas totales:** {len(df_reporte)}\n\n")
            f.write("## 📌 Top 10 Alertas de Fraude\n\n")
            f.write("| Barrio | Uso | Fecha | Riesgo (%) | Nivel | Motivo |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for _, row in df_reporte.head(10).iterrows():
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