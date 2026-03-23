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

    @staticmethod
    def find_optimal_clusters(X_sequences: np.ndarray, max_clusters=10, feature_idx=0) -> int:
        """
        Ejecuta el Método del Codo (Elbow Method) para determinar matemáticamente 
        el número óptimo de clústeres sin necesidad de gráficas.
        """
        print(f"Buscando el número óptimo de clústeres (hasta k={max_clusters})...")
        
        # 1. Extraer y escalar la feature
        X_target = X_sequences[:, :, feature_idx:feature_idx+1]
        X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_target)
        
        # 2. Calcular inercia para cada K
        K = list(range(1, max_clusters + 1))
        inertias = []
        
        for k in K:
            km = TimeSeriesKMeans(n_clusters=k, metric="dtw", n_jobs=-1, random_state=AIConstants.RANDOM_STATE)
            inertias.append(km.fit(X_scaled).inertia_)
            print(f"  > Evaluando k={k:02d} | Inercia: {inertias[-1]:.2f}")
            
        # 3. Detectar matemáticamente el codo (mayor distancia a la recta entre extremos)
        m = (inertias[-1] - inertias[0]) / (K[-1] - K[0])  # Pendiente de la recta
        c = inertias[0] - m * K[0]                         # Intersección
        
        distancias = [abs(inertias[i] - (m * K[i] + c)) for i in range(len(K))]
        mejor_k = K[np.argmax(distancias)]
        
        # 4. Mostrar resultados limpios
        print("\n" + "="*40)
        print(f"EL NÚMERO ÓPTIMO DE CLÚSTERES ES: {mejor_k}")
        print("="*40 + "\n")
        
        return mejor_k
        
    @staticmethod
    def plot_cluster_samples(X_sequences: np.ndarray, labels: np.ndarray, 
                            image_path: Path, feature_idx=0, fig_size=(16, 12)) -> None:
        """
        Visualización óptima en dos paneles:
        1. Arriba: Perfil de consumo PROMEDIO (Centroide) del clúster con su desviación.
        2. Abajo: Boxplots agrupados por mes para ver la dispersión real.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        
        # 1. ESTÉTICA: Tema limpio y profesional
        sns.set_theme(style="whitegrid", context="notebook")
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        palette = sns.color_palette("husl", n_clusters)
        meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        
        # 2. PREPARACIÓN DE DATOS (Panel inferior)
        data = []
        for i in range(len(X_sequences)):
            cluster_id = labels[i]
            for mes_idx in range(12):
                consumo = X_sequences[i, mes_idx, feature_idx]
                data.append({
                    "Cluster": f"Clúster {cluster_id}",
                    "Mes": meses_nombres[mes_idx],
                    "Consumo": consumo
                })
        df_plot = pd.DataFrame(data)
        
        # 3. CREACIÓN DE LA FIGURA
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [1, 1.5]})
        fig.suptitle("Análisis de Clústeres: Perfil Promedio y Dispersión Mensual", fontsize=18, fontweight='bold', y=0.98)
        
        # ---------------------------------------------------------
        # PANEL 1 (ARRIBA): CENTROIDE (Media) + Desviación Estándar
        # ---------------------------------------------------------
        for i, c in enumerate(unique_labels):
            idx_c = np.where(labels == c)[0]
            if len(idx_c) > 0:
                # Extraemos TODAS las secuencias de este clúster
                seqs = X_sequences[idx_c, :, feature_idx]
                
                # Calculamos la media y la desviación estándar por mes
                mean_seq = np.mean(seqs, axis=0)
                std_seq = np.std(seqs, axis=0)
                
                # Pintamos la línea de la media (El patrón real del clúster)
                ax1.plot(meses_nombres, mean_seq, label=f"Clúster {c} (n={len(idx_c)})", color=palette[i], linewidth=3, marker='o')
                
                # Pintamos el área sombreada (La varianza de los datos)
                ax1.fill_between(meses_nombres, 
                                 np.maximum(0, mean_seq - std_seq), # Evitamos que baje de 0 visualmente
                                 mean_seq + std_seq, 
                                 color=palette[i], alpha=0.15)
                
        ax1.set_title("1. Patrón de Consumo Promedio por Clúster (Sombreado = Variabilidad)", fontsize=14)
        ax1.set_ylabel("Consumo Promedio [0-1]", fontsize=12)
        ax1.legend(loc='upper right')
        
        # ---------------------------------------------------------
        # PANEL 2 (ABAJO): Distribución completa (Boxplot)
        # ---------------------------------------------------------¡    
        sns.boxplot(
            data=df_plot,
            x="Mes",
            y="Consumo",
            hue="Cluster",
            palette=palette,
            fliersize=2,       
            linewidth=1,
            ax=ax2
        )
        ax2.set_title("2. Dispersión Real (Outliers incluidos)", fontsize=14)
        ax2.set_ylabel("Consumo Escala [0-1]", fontsize=12)
        ax2.set_xlabel("Meses del Año", fontsize=12)
        
        ax2.get_legend().remove() 
        
        plt.tight_layout()
        if image_path:
            plt.savefig(image_path / "clustering.png")
        plt.show()
        
        sns.reset_orig()