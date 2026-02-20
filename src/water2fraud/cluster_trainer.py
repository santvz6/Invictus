import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import get_logger, DatasetKeys
from src.water2fraud.models.water_segmenter import WaterSegmenter
from src.water2fraud.features.preprocessor import WaterPreprocessor

logger = get_logger(__name__)

class WaterClusterTrainer:
    def __init__(self, model: WaterSegmenter, preprocessor: WaterPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.data_processed = None
        self.results = None

    def run_pipeline(self, df_raw):
        """Ejecuta el flujo completo: Preprocesar -> Entrenar -> Predecir"""
        logger.info("Iniciando Pipeline de Entrenamiento...")
        
        # 1. Preprocesamiento
        self.data_processed = self.preprocessor.create_feature_matrix(df_raw)

        # 2. Entrenamiento
        self.model.train(self.data_processed)
       
        X_scaled = self.model.scaler.transform(self.data_processed)
        total_variance = np.sum(np.var(X_scaled, axis=0))
        explicatividad = (1 - (self.model.wcss / (total_variance * len(X_scaled)))) * 100
        logger.info(f"La IA explica el {explicatividad:.2f}% de la variabilidad de los datos.")

        # 3. Asignación de Clusters
        labels = self.model.predict(self.data_processed)
        self.results = self.data_processed.copy()
        self.results[DatasetKeys.CLUSTER] = labels
        
        logger.info("Pipeline finalizado con éxito.")
        return self.results

    def save_artifacts(self, output_path):
        """Guarda el modelo y las gráficas de los centroides"""
        output_path.mkdir(parents=True, exist_ok=True)

        # Model
        model_filename = output_path / "water_segmenter_model.joblib"
        try:
            joblib.dump(self.model, model_filename)
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")

        # Plots
        centroids = self.model.get_centroids()
        self._plot_centroids_heatmap(centroids, output_path)
        self._plot_clusters_scatter(output_path)

    def _plot_centroids_heatmap(self, centroids, path):
        """Heatmap legible con etiquetas del Schema"""
        df_centroids = pd.DataFrame(centroids, columns=DatasetKeys.get_feature_columns())
        
        plt.figure(figsize=(16, 6))
        sns.heatmap(df_centroids, annot=True, fmt=".3f", cmap="YlGnBu", robust=True)
        plt.title("Huella Hídrica de los Centroides (Promedio por Cluster)")
        plt.xlabel("Características")
        plt.ylabel("ID Cluster")
        plt.tight_layout()
        plt.savefig(path / "centroids_heatmap.png")
        plt.close()

    def _plot_clusters_scatter(self, path):
        """Muestra los puntos reales agrupados (Visión de Negocio)"""
        plt.figure(figsize=(10, 7))
        
        # Graficamos Consumo Medio vs Ratio Fin de Semana
        sns.scatterplot(data=self.results, x=DatasetKeys.MEAN_CONSUMO, y=DatasetKeys.RATIO_WEEKEND, 
                        hue=DatasetKeys.CLUSTER, palette="viridis", alpha=0.6, edgecolor="w", s=100
        )
        
        plt.title("Segmentación de Usuarios: Volumen vs Estacionalidad")
        plt.xlabel("Consumo Medio Diario (m3)")
        plt.ylabel("Ratio Fin de Semana (Weekend/Weekday)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(path / "clusters_distribution.png")
        plt.close()