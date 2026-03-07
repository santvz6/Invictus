import numpy as np
import joblib

from pathlib import Path
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from src.config import AIConstants

class ClusterManager:
    """
    Gestor para la agrupación (clustering) de series temporales.
        
    Utiliza K-Means adaptado para series temporales con la métrica Dynamic Time Warping (DTW),
    lo que permite agrupar curvas de consumo basándose en la similitud de su forma, independientemente
    de pequeños desfases temporales.
    """
    def __init__(self, n_clusters=AIConstants.N_CLUSTERS_DEFAULT) -> None:
        """
        Inicializa el modelo de clustering y el escalador.

        Args:
            n_clusters (int, optional): Número de clústeres a formar. Por defecto toma el valor
                                        definido en AIConstants.N_CLUSTERS_DEFAULT.
        """
        self.n_clusters = n_clusters
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters, 
            metric="dtw", 
            n_jobs=-1, # Usamos todos los núcleos
            random_state=AIConstants.RANDOM_STATE
        )
        self.scaler = TimeSeriesScalerMeanVariance()

    def fit_predict(self, X_sequences: np.ndarray, feature_idx=0) -> np.ndarray:
        """
        Escala las series temporales y asigna cada una a un clúster.

        Args:
            X_sequences (np.ndarray): Array 3D con forma (N, seq_len, features).
            feature_idx (int, optional): Índice de la característica a utilizar para el clustering.
                                         Por defecto es 0 (CONTRATO_RATIO).

        Returns:
            np.ndarray: Array unidimensional con las etiquetas de clúster asignadas a cada muestra.
        """
        # Extraemos solo la serie temporal objetivo (ej: Consumo Ratio)
        # Forma original: (N, seq_len, features) -> Extraemos (N, seq_len, 1)
        X_target = X_sequences[:, :, feature_idx].reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        # Escalamos para agrupar por "forma de la curva" y no por volumen total
        X_scaled = self.scaler.fit_transform(X_target)
        
        labels = self.model.fit_predict(X_scaled)
        return labels

    def save(self, path: Path) -> None:
        """
        Serializa y guarda el modelo de clustering entrenado en disco.

        Args:
            path (str o Path): Ruta de destino donde se guardará el modelo (ej. formato .joblib).
        """
        joblib.dump(self.model, path)