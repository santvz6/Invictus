from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import AIConstants, get_logger

logger = get_logger(__name__)

class WaterSegmenter:
    """
    FASE 1: ...
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=AIConstants.RANDOM_STATE, 
            n_init=AIConstants.KMEANS_N_INIT
        )

    def train(self, X):
        logger.info(f"Entrenando K-Means con k={self.n_clusters}")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        self.wcss = self.model.inertia_
        logger.info(f"Entrenamiento completado. WCSS (Inertia): {self.wcss:.2f}")
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_centroids(self):
        # Devolvemos los centros en la escala original para la Fase 2
        return self.scaler.inverse_transform(self.model.cluster_centers_)