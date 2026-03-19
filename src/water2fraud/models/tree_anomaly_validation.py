import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.config import AIConstants

logger = logging.getLogger(__name__)

class TreeModelValidator:
    """
    Suite de modelos de Machine Learning clásico para validar y complementar 
    al Autoencoder en la detección de anomalías.
    """

    @staticmethod
    def flatten_sequences(X_3d: np.ndarray) -> np.ndarray:
        """
        Convierte el tensor 3D de secuencias temporales en una matriz 2D tabular.
        Forma original: (Muestras, 12_meses, Num_Features)
        Forma nueva: (Muestras, 12_meses * Num_Features)
        """
        nsamples, n_seq, n_features = X_3d.shape
        return X_3d.reshape((nsamples, n_seq * n_features))

    @staticmethod
    def run_isolation_forest(X_2d: np.ndarray, contamination: float = 0.05):
        """
        Modelo No Supervisado: Busca aislar viviendas raras haciendo divisiones aleatorias.
        """
        logger.info("Entrenando Isolation Forest...")
        # contamination = porcentaje estimado de fraudes (ej. 5%)
        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=AIConstants.RANDOM_STATE)
        
        # Entrenamos y predecimos
        preds = iso.fit_predict(X_2d)
        
        # Isolation forest devuelve -1 para anomalías y 1 para normales. 
        # Lo pasamos a booleanos (True = Fraude)
        anomalies_bool = preds == -1 
        return anomalies_bool, iso

    @staticmethod
    def run_surrogate_models(X_2d: np.ndarray, y_labels: np.ndarray):
        """
        Modelos Supervisados: Entrenamos RF y XGBoost para intentar replicar 
        las decisiones del Autoencoder.
        """
        logger.info("Entrenando Random Forest y XGBoost como modelos sustitutos...")
        
        # Como las anomalías son minoría (desbalanceo de clases), usamos class_weight
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=AIConstants.RANDOM_STATE, n_jobs=-1)
        
        # XGBoost maneja el desbalanceo con scale_pos_weight
        num_negatives = np.sum(y_labels == 0)
        num_positives = np.sum(y_labels == 1)
        # Protegemos contra división por cero si no hay anomalías
        peso_positivo = num_negatives / num_positives if num_positives > 0 else 1
        
        xgb = XGBClassifier(n_estimators=100, scale_pos_weight=peso_positivo, random_state=AIConstants.RANDOM_STATE, eval_metric='logloss')

        # Entrenamos
        rf.fit(X_2d, y_labels)
        xgb.fit(X_2d, y_labels)

        # Predicciones
        rf_preds = rf.predict(X_2d)
        xgb_preds = xgb.predict(X_2d)

        return rf, xgb, rf_preds, xgb_preds

    @staticmethod
    def evaluate_surrogate(y_true, y_pred, model_name="Modelo"):
        """Imprime un reporte rápido de cómo de bien el árbol replicó a la IA."""
        print(f"\\n--- Resultados de {model_name} frente al Autoencoder ---")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))

    @staticmethod
    def plot_isolation_forest_results(X_2d: np.ndarray, iso_model: IsolationForest, anomalies_bool: np.ndarray, uso_labels: np.ndarray = None):
        """
        Genera una visualización doble para entender el comportamiento del Isolation Forest.
        Permite clasificar los puntos del PCA mediante la etiqueta 'Uso'.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        
        scores = iso_model.decision_function(X_2d)
        
        # Hacemos el lienzo un poco más ancho para que quepa la nueva leyenda
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        plt.style.use('dark_background')
        
        # --- GRÁFICO 1: Distribución del Score ---
        sns.histplot(scores[~anomalies_bool], bins=50, color='#3498db', label='Normales', kde=True, ax=axes[0])
        sns.histplot(scores[anomalies_bool], bins=10, color='#e74c3c', label='Anomalías (Fraude)', kde=True, ax=axes[0])
        
        corte = scores[anomalies_bool].max()
        axes[0].axvline(corte, color='white', linestyle='--', linewidth=2, label='Umbral de Corte')
        
        axes[0].set_title('Distribución de Puntuaciones de Anomalía', fontsize=14)
        axes[0].set_xlabel('Anomaly Score (Más negativo = Más anómalo)')
        axes[0].set_ylabel('Número de Viviendas')
        axes[0].legend()

        # --- GRÁFICO 2: Proyección PCA 2D por Categoría ---
        logger.info("Calculando PCA para visualización 2D...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_2d)
        
        if uso_labels is not None:
            # Diccionario de colores para los tipos de Uso
            colores_uso = {
                'DOMESTICO': '#2ecc71',    # Verde esmeralda
                'COMERCIAL': '#f39c12',    # Naranja
                'NO DOMESTICO': '#9b59b6'  # Morado
            }
            
            usos_unicos = np.unique(uso_labels)
            
            for uso in usos_unicos:
                # Nos aseguramos de cruzar el uso con nuestro diccionario (por si hay variaciones de texto)
                color = colores_uso.get(uso.upper().strip(), '#ffffff') 
                mask_uso = uso_labels == uso
                
                # 1. Puntos Normales de este Uso (Círculos pequeños y transparentes)
                mask_normal = mask_uso & (~anomalies_bool)
                if np.any(mask_normal):
                    axes[1].scatter(X_pca[mask_normal, 0], X_pca[mask_normal, 1], 
                                    c=color, alpha=0.3, s=20, marker='o', label=f'Normal - {uso}')
                
                # 2. Anomalías de este Uso (Equis grandes, sólidas y con borde blanco)
                mask_anomalia = mask_uso & anomalies_bool
                if np.any(mask_anomalia):
                    axes[1].scatter(X_pca[mask_anomalia, 0], X_pca[mask_anomalia, 1], 
                                    c=color, alpha=1.0, s=120, marker='X', edgecolor='white', 
                                    linewidths=1.5, label=f'🚨 Fraude - {uso}')
        else:
            # Fallback por si alguien llama a la función sin pasar la variable 'uso_labels'
            axes[1].scatter(X_pca[~anomalies_bool, 0], X_pca[~anomalies_bool, 1], 
                            c='#3498db', alpha=0.4, s=20, label='Consumo Normal')
            axes[1].scatter(X_pca[anomalies_bool, 0], X_pca[anomalies_bool, 1], 
                            c='#e74c3c', alpha=0.9, s=80, marker='X', edgecolor='white', label='🚨 Anomalía Detectada')
        
        axes[1].set_title('Mapa Físico de Anomalías (PCA) por Tipo de Uso', fontsize=14)
        axes[1].set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1].set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        
        # Movemos la leyenda fuera del gráfico para que no tape los puntos
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()