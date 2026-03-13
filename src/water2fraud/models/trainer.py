import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class EarlyStopping:
    """
    Detiene el entrenamiento si la métrica de pérdida no mejora tras un número de épocas.
        """
    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience (int): Número de épocas a esperar sin mejora antes de detener.
            min_delta (float): Cambio mínimo para ser considerado una mejora.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Guardamos una copia profunda de los mejores pesos
            self.best_model_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_autoencoder(model: nn.Module, dataloader: DataLoader, epochs=100, lr=1e-3, 
                      patience=15, device="cpu") -> tuple[nn.Module, dict]:
    """
    Bucle de entrenamiento estandarizado para el Autoencoder LSTM con Early Stopping.

    Args:
        model (nn.Module): Instancia de la red neuronal a entrenar.
        dataloader (DataLoader): Iterador de PyTorch con los datos.
        epochs (int): Número máximo de épocas.
        lr (float): Tasa de aprendizaje (Learning Rate).
        patience (int): Épocas de tolerancia para el Early Stopping.
        device (str): Dispositivo ('cpu' o 'cuda').

    Returns:
        tuple: (Modelo entrenado con los mejores pesos, Diccionario con el historial de pérdidas)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() # MAE
    
    early_stopping = EarlyStopping(patience=patience)
    history = {'loss': []}
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            reconstruction = model(batch_x)
            
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        history['loss'].append(avg_loss)
        
        # Comprobación de Early Stopping
        early_stopping(avg_loss, model)
        
        if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.5f}")
            
        if early_stopping.early_stop:
            print(f"  -> Early Stopping activado en la época {epoch+1}. Restaurando mejores pesos.")
            break
            
    # Restauramos el modelo a su mejor estado antes del overfitting
    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
        
    return model, history


def detect_anomalies(model: nn.Module, X_sequences: np.ndarray, metadata_df: pd.DataFrame, 
                     physics_threshold=1.5, device="cpu") -> pd.DataFrame:
    """
    Evalúa secuencias mediante el Autoencoder y aplica reglas físicas para detectar anomalías.
    
    Calcula el error de reconstrucción de cada serie. Si el error supera un umbral
    estadístico (percentil 95) Y el consumo real supera el consumo teórico dictado por
    las fórmulas físicas, marca la muestra como un fraude o vivienda turística ilegal.

    Args:
        model (nn.Module): Autoencoder LSTM previamente entrenado.
        X_sequences (np.ndarray): Secuencias a evaluar con forma (N, seq_len, num_features).
        metadata_df (pd.DataFrame): Metadatos asociados a las secuencias (Barrio, Fecha, consumos).
        physics_threshold (float, optional): Multiplicador del límite físico de consumo teórico. 
                                             Por defecto es 1.5.
        device (str, optional): Dispositivo de cómputo ('cpu' o 'cuda'). Por defecto es 'cpu'.

    Returns:
        pd.DataFrame: DataFrame con los metadatos originales extendidos con los errores de
                      reconstrucción, flags de anomalías y la alerta final.
    """
    model.eval()
    model.to(device)
    
    X_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        reconstructions = model(X_tensor)
        
    # Calculamos el error absoluto (MAE) por cada secuencia
    # Dim 1: seq_len, Dim 2: features
    errors = torch.mean(torch.abs(reconstructions - X_tensor), dim=(1,2)).cpu().numpy()
    
    # 1. Obtenemos el umbral estadístico del Autoencoder (ej: percentil 95 de error)
    ae_threshold = np.percentile(errors, 95)
    
    results = metadata_df.copy()
    results['reconstruction_error'] = errors
    results['is_ae_anomaly'] = errors > ae_threshold
    
    # 2. POST-PROCESAMIENTO CON LA FÍSICA
    # Asumimos que tienes el consumo real vs teórico en el metadata_df
    # (Lo habremos cruzado previamente al armar el dataset original)
    if 'consumo_real' in results.columns and 'consumo_fisico_teorico' in results.columns:
        
        # Condición Física: ¿Supera la realidad a la teoría por más de X veces?
        results['is_physics_anomaly'] = results['consumo_real'] > (results['consumo_fisico_teorico'] * physics_threshold)
        
        # ALERTA DE VIVIENDA ILEGAL (Ambos modelos concuerdan)
        results['ALERTA_TURISTICA_ILEGAL'] = results['is_ae_anomaly'] & results['is_physics_anomaly']
        
    return results


# =====================================================================
# HERRAMIENTAS DE VISUALIZACIÓN PARA JUPYTER NOTEBOOKS
# =====================================================================

def plot_training_history(history: dict, title="Historial de Entrenamiento del Autoencoder"):
    """Dibuja la curva de pérdida durante el entrenamiento."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss (MAE)', color='blue', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_reconstruction(model: nn.Module, sequence: np.ndarray, feature_idx=0, 
                        feature_name="Consumo", device="cpu"):
    """
    Compara visualmente la secuencia de entrada original con la reconstrucción del Autoencoder.
    Ideal para ver en el Notebook cómo falla el modelo ante un fraude.
    """
    model.eval()
    model.to(device)
    
    # Preparamos el tensor (1 muestra, seq_len, features)
    seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstruction = model(seq_tensor).squeeze(0).cpu().numpy()
        
    original = sequence[:, feature_idx]
    reconst = reconstruction[:, feature_idx]
    
    plt.figure(figsize=(10, 4))
    plt.plot(original, label=f'Original ({feature_name})', marker='o', color='black')
    plt.plot(reconst, label=f'Reconstrucción ({feature_name})', marker='x', color='red', linestyle='--')
    
    mae = np.mean(np.abs(original - reconst))
    plt.title(f"Reconstrucción del Autoencoder | MAE: {mae:.4f}", fontsize=14)
    plt.xlabel("Meses de la secuencia")
    plt.ylabel("Valor escalado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()