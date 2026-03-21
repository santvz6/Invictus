import torch.nn as nn
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
from src.config import get_logger

logger = get_logger(__name__)

class AEErrorWrapper(nn.Module):
    """
    Cápsula para el Autoencoder. 
    En lugar de devolver la matriz reconstruida (Tensor 3D), devuelve el 
    Error Absoluto Medio (Tensor 1D) para cada vivienda. 
    Esto permite a SHAP calcular gradientes escalares sin que PyTorch colapse.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        reconstruction = self.model(x)
        # Calculamos el MAE de cada vivienda promediando meses (dim 1) y features (dim 2)
        # Forma resultante: (batch_size, 1)
        error = torch.mean(torch.abs(reconstruction - x), dim=[1, 2], keepdim=True)
        return error


class ModelExplainer:
    
    @staticmethod
    def plot_feature_importance(model, X_train_tensor, X_anomalies_tensor, feature_names, device="cpu", save_path=None):
        logger.info("Calculando valores SHAP para interpretabilidad...")
        
        # 🔇 SILENCIAMOS EL WARNING MOLESTO DE SHAP Y NUMPY
        warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")
        
        cudnn_enabled_original = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        
        try:
            model.eval()
            
            wrapper_model = AEErrorWrapper(model).to(device)
            wrapper_model.eval()
            
            num_background = min(100, X_train_tensor.shape[0])
            background_indices = np.random.choice(X_train_tensor.shape[0], num_background, replace=False)
            background = X_train_tensor[background_indices].to(device)
            
            anomalies = X_anomalies_tensor.to(device)
            
            explainer = shap.GradientExplainer(wrapper_model, background)
            shap_values = explainer.shap_values(anomalies)
            
            if isinstance(shap_values, list):
                shap_values_3d = shap_values[0]
            else:
                shap_values_3d = shap_values
                
            # 🌟 CORRECCIÓN MATEMÁTICA VITAL
            # Sumamos el impacto real a lo largo de los 12 meses
            shap_values_2d = np.sum(shap_values_3d, axis=1)
            
            # 🚀 EL FIX: Aplastamos la dimensión fantasma (pasamos de 201x7x1 a 201x7)
            shap_values_2d = np.squeeze(shap_values_2d) 
            
            X_anomalies_2d = np.mean(anomalies.cpu().numpy(), axis=1)
            
            # Debug de seguridad para tu consola
            logger.info(f"Dimensiones de los valores SHAP: {shap_values_2d.shape}")
            logger.info(f"Variables detectadas: {len(feature_names)}")
            
            # 🌟 CORRECCIÓN VISUAL
            # Dejamos que SHAP gestione el tamaño del lienzo internamente
            # 1. Generamos el gráfico base con transparencia y mejor mapa de colores
            shap.summary_plot(
                shap_values_2d, 
                X_anomalies_2d, 
                feature_names=feature_names, 
                plot_size=(11, 7), 
                alpha=0.5,           # Transparencia: si hay muchos puntos solapados, se verá más intenso
                cmap=plt.get_cmap("coolwarm"), # Paleta de colores más elegante (azul a rojo intenso)
                show=False
            )
            
            # 2. Recuperamos el lienzo para hacer magia con Matplotlib
            ax = plt.gca()
            
            # 3. LA CLAVE: Escala logarítmica para evitar el aplastamiento
            # Expande los valores cercanos a 0 y comprime los valores gigantes.
            # (El linthresh=0.01 define dónde empieza a actuar el logaritmo)
            ax.set_xscale('symlog', linthresh=0.05)
            
            # 4. Limpieza estética (Tufte style)
            ax.grid(True, axis='x', linestyle='--', alpha=0.4, color='gray') # Rejilla vertical sutil
            ax.set_facecolor('#fdfdfd') # Un fondo sutilmente gris/blanco para dar profundidad
            
            # Ocultamos los bordes superior y derecho para dar un aspecto más limpio y moderno
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # 5. Títulos y etiquetas descriptivas
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
            torch.backends.cudnn.enabled = cudnn_enabled_original