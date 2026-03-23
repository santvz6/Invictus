import shap
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import AIConstants, get_logger


logger = get_logger(__name__)

class AEErrorWrapper(nn.Module):
    """
    Cápsula para el Autoencoder. 
    En lugar de devolver la matriz reconstruida (Tensor 3D), devuelve el 
    Error Absoluto Medio (Tensor 1D) para cada vivienda. 
    Esto permite a SHAP calcular gradientes escalares sin que PyTorch colapse.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        reconstruction = self.model(x)
        # Calculamos el MAE de cada vivienda promediando meses (dim 1) y features (dim 2)
        # Forma resultante: (batch_size, 1)
        error = torch.mean(torch.abs(reconstruction - x), dim=[1, 2], keepdim=True)
        return error


class SHAPAutoencoderExplainer:
    """
    Clase encargada de la interpretabilidad del modelo de detección de anomalías.
    Utiliza SHAP (SHapley Additive exPlanations) para explicar por qué una vivienda
    ha sido marcada como sospechosa, identificando qué variables y meses
    contribuyeron más al error de reconstrucción del Autoencoder.
    """
    
    @staticmethod
    def plot_feature_importance(model: nn.Module, X_train_tensor: np.ndarray, X_anomalies_tensor: np.ndarray, 
                                feature_names: list[str], device: str = "cpu", save_path: str = None) -> None:
        """
        Genera un gráfico de importancia de características utilizando SHAP.
        
        Args:
            model (nn.Module): Modelo Autoencoder entrenado.
            X_train_tensor (np.ndarray): Tensor de NumPy con las secuencias de entrenamiento.
            X_anomalies_tensor (np.ndarray): Tensor de NumPy con las secuencias anómalas.
            feature_names (list[str]): Lista con los nombres de las características.
            device (str): Dispositivo para realizar la inferencia ('cpu' o 'cuda').
            save_path (str): Ruta para guardar el gráfico.
        """
        logger.info("Calculando valores SHAP para interpretabilidad...")
        
        # Silenciamos el warning de numpy
        warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")
        
        # Desactivamos cudnn para evitar errores con SHAP
        cudnn_enabled_original = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        
        try:
            model.eval()
            
            # Envolvemos el modelo para que devuelva el error en lugar de la reconstrucción
            wrapper_model = AEErrorWrapper(model).to(device)
            wrapper_model.eval()
            
            # Seleccionamos un subconjunto de datos de entrenamiento para el cálculo de SHAP
            num_background = min(100, X_train_tensor.shape[0])
            background_indices = np.random.choice(X_train_tensor.shape[0], num_background, replace=False)
            background = X_train_tensor[background_indices].to(device)
            
            # Obtenemos las secuencias anómalas
            anomalies = X_anomalies_tensor.to(device)
            
            # Calculamos los valores SHAP
            explainer = shap.GradientExplainer(wrapper_model, background)
            shap_values = explainer.shap_values(anomalies)
            
            # Verificamos si los valores SHAP son una lista
            if isinstance(shap_values, list):
                shap_values_3d = shap_values[0]
            else:
                shap_values_3d = shap_values
                
            # CORRECCIÓN MATEMÁTICA VITAL
            # Sumamos el impacto real a lo largo de los 12 meses
            shap_values_2d = np.sum(shap_values_3d, axis=1)
            
            # Aplastamos la dimensión fantasma (pasamos de 201x7x1 a 201x7)
            shap_values_2d = np.squeeze(shap_values_2d) 
        
            X_anomalies_2d = np.mean(anomalies.cpu().numpy(), axis=1)
     
            # Generamos el gráfico base con transparencia y mejor mapa de colores
            shap.summary_plot(
                shap_values_2d, 
                X_anomalies_2d, 
                feature_names=feature_names, 
                plot_size=(11, 7), 
                alpha=0.5,                      # Transparencia: si hay muchos puntos solapados, se verá más intenso
                cmap=plt.get_cmap("coolwarm"),
                show=False
            )
            
            ax = plt.gca()
            
            # Escala logarítmica para evitar el aplastamiento
            # Expande los valores cercanos a 0 y comprime los valores gigantes.
            # (El linthresh=0.01 define dónde empieza a actuar el logaritmo)
            ax.set_xscale('symlog', linthresh=0.05)
            
            # Limpieza estética (Tufte style)
            ax.grid(True, axis='x', linestyle='--', alpha=0.4, color='gray') # Rejilla vertical sutil
            ax.set_facecolor('#fdfdfd') # Un fondo sutilmente gris/blanco
            
            # Ocultamos los bordes superior y derecho
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Títulos y etiquetas descriptivas
            ax.set_title("Auditoría de IA: Impacto de variables en la Detección de Fraude", 
                         fontsize=16, fontweight='bold', pad=20, color='#333333')
            ax.set_xlabel("Impacto SHAP (Escala Logarítmica - Aumento del Error)", 
                          fontsize=12, fontweight='medium', color='#555555')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Gráfico SHAP guardado en: {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        finally:
            # Restauramos el estado original de cudnn
            torch.backends.cudnn.enabled = cudnn_enabled_original


class TreeAutoencoderExplainer:
    """
    Clase encargada de entrenar modelos de árboles de decisión para explicar 
    los patrones de reconstrucción o las representaciones latentes de un Autoencoder.
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
        Entrena modelos subrogados para replicar las etiquetas generadas por el Autoencoder.
        Utiliza una división Train/Test para validar qué tan fiel es el árbol al modelo original.
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("Entrenando modelos con separación Train/Test...")
        
        # 1. Separamos los datos (80% entrenamiento, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_2d, y_labels, test_size=0.2, random_state=AIConstants.RANDOM_STATE, stratify=y_labels
        )
        
        # 2. Calculamos el peso solo con el tren de entrenamiento
        num_negatives = np.sum(y_train == 0)
        num_positives = np.sum(y_train == 1)
        peso_positivo = num_negatives / num_positives if num_positives > 0 else 1
        
        # 3. Configuramos modelos
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                    random_state=AIConstants.RANDOM_STATE, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=100, scale_pos_weight=peso_positivo, 
                            random_state=AIConstants.RANDOM_STATE, eval_metric='logloss')

        # 4. Entrenamos con TRAIN
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        # 5. Predecimos sobre TEST (aquí es donde se ve la verdad)
        rf_preds = rf.predict(X_test)
        xgb_preds = xgb.predict(X_test)

        # Retornamos los modelos y las etiquetas reales de test para evaluar
        return rf, xgb, rf_preds, xgb_preds, y_test
    
    @staticmethod
    def evaluate_surrogate(y_true, y_pred, model_name, output_path):
        """Imprime un reporte rápido de cómo de bien el árbol replicó a la IA."""
        print(f"\\n--- Resultados de {model_name} frente al Autoencoder ---")
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # 'd' para números enteros
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        save_file = output_path / f"{model_name}_cm.png"
        plt.savefig(save_file)
        plt.close()

        print(cm)
        print(classification_report(y_true, y_pred))


    @staticmethod
    def plot_tree_importance(tree_model, feature_names: list[str], output_path: Path):
        """
        Calcula la importancia acumulada. Dado que los datos fueron aplanados (12 meses * N features),
        esta función agrupa la importancia de una misma variable a lo largo de todo el año.
        """

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
        plt.savefig(output_path / f"{tree_model.__class__.__name__}_importances.png")
        plt.show()