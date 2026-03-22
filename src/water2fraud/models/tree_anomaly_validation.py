import numpy as np
import pandas as pd
import logging

import matplotlib.pyplot as plt
import seaborn as sns

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
    def evaluate_surrogate(y_true, y_pred, model_name, output_path):
        """Imprime un reporte rápido de cómo de bien el árbol replicó a la IA."""
        print(f"\\n--- Resultados de {model_name} frente al Autoencoder ---")
        cm = confusion_matrix(y_true, y_pred)
        plt.savefig(output_path / f"{model_name}_cm.png")
        print(cm)
        print(classification_report(y_true, y_pred))


    @staticmethod
    def plot_tree_importance(tree_model, feature_names):
        # 1. Calculamos la importancia agrupada en una sola línea (Vectorizado)
        importancias = tree_model.feature_importances_
        # Sumamos cada n variables (asumiendo que feature_names tiene el tamaño original, ej. 7)
        n = len(feature_names)
        imp_agrupada = [sum(importancias[i::n]) for i in range(n)]

        # 2. Creamos y ordenamos el DataFrame
        df_imp = pd.DataFrame({'Variable': feature_names, 'Importancia': imp_agrupada})
        df_imp = df_imp.sort_values('Importancia', ascending=False)

        # 3. Plot
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_imp, x='Importancia', y='Variable', hue='Variable', palette='viridis', legend=False)
        
        plt.title(f"Variables clave: {tree_model.__class__.__name__}")
        plt.xlabel("Importancia Acumulada")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()