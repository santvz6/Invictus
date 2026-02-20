from src.config import get_logger, BusinessLabels, DatasetKeys
import numpy as np

logger = get_logger(__name__)

class WaterLabeler:
    """
    FASE 2: Orquestador de lógica de negocio para la Fase 2 del pipeline.
    
    Esta clase se encarga de transformar el conocimiento matemático (centroides)
    en categorías operativas aplicando umbrales físicos sobre el comportamiento 
    del consumo hídrico.
    """
    @staticmethod
    def define_labels(centroids:np.ndarray, feature_names:list[str]) -> dict[str, str]:
        """
        Asigna una etiqueta semántica a cada cluster basada en su perfil medio.

        Args:
            centroids (np.ndarray): Matriz de centros de forma (n_clusters, n_features)
                en escala real (m3/ratios), no normalizada.
            feature_names (list[str]): Lista ordenada de nombres de las columnas 
                que componen los centroides (H0...H23, ratio_wk, mean, std).

        Returns:
            dict[str, str]: Diccionario de mapeo donde la clave es el ID del cluster 
                (como string) y el valor es la etiqueta de BusinessLabels.
        """
        labels : dict[str, str] = {}
        
        # Mapeo de índices para no depender del orden
        idxs_hourly : list[int] = [feature_names.index(f"H{i}") for i in range(24)]
        idx_wk      : int = feature_names.index(DatasetKeys.RATIO_WEEKEND)
        idx_std     : int = feature_names.index(DatasetKeys.STD_CONSUMO)
        idx_mean    : int = feature_names.index(DatasetKeys.MEAN_CONSUMO)

        for i, center in enumerate(centroids):
            hourly_ratios  : list[float] = center[idxs_hourly] # 1 x 27
            wk_ratio       : float       = center[idx_wk]
            std_val        : float       = center[idx_std]
            mean_val       : float       = center[idx_mean]
            
            """
            Si quieres acceder al ratio de H7:
            
            >>> h7_ratio: float = hourly_ratios[7]
            """

            # Modelado Matemático / Físico
            # TODO: Mejorar el modelado
            if wk_ratio > 1.4:
                labels[str(i)] = BusinessLabels.TURISTICO
            elif mean_val < 5e-3:
                labels[str(i)] = BusinessLabels.DESOCUPADO
            elif std_val < 1e-3 and mean_val > 5e-2:
                labels[str(i)] = BusinessLabels.INDUSTRIAL_FUGA
            else:
                labels[str(i)] = BusinessLabels.RESIDENCIAL
            
            logger.info(f"Cluster {i} etiquetado como: {labels[str(i)]} (Ratio: {wk_ratio:.2f})")
            
        return labels