import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DatasetKeys, AIConstants

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


def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader = None, 
                      epochs=100, lr=1e-3, patience=15, device="cpu") -> tuple[nn.Module, dict]:
    """
    Bucle de entrenamiento mejorado para el Autoencoder LSTM con Validación y Early Stopping.

    Args:
        model (nn.Module): Instancia de la red neuronal a entrenar.
        train_loader (DataLoader): Iterador para los datos de entrenamiento.
        val_loader (DataLoader, optional): Iterador para los datos de validación.
        epochs (int): Número máximo de épocas.
        lr (float): Tasa de aprendizaje inicial.
        patience (int): Épocas de tolerancia para el Early Stopping.
        device (str): Dispositivo ('cpu' o 'cuda').

    Returns:
        tuple: (Modelo entrenado con los mejores pesos, Diccionario con el historial de pérdidas)
    """
    model.to(device)
    # Cambiamos a AdamW para un mejor manejo del weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Scheduler para reducir el LR cuando la pérdida se estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    criterion = nn.L1Loss() # MAE
    
    early_stopping = EarlyStopping(patience=patience)
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            reconstruction = model(batch_x)
            
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            
            # Gradient Clipping para estabilizar LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        history['loss'].append(avg_train_loss)
        
        # --- FASE DE VALIDACIÓN ---
        avg_val_loss = avg_train_loss
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_val, _ in val_loader:
                    batch_val = batch_val.to(device)
                    reconstruction = model(batch_val)
                    loss = criterion(reconstruction, batch_val)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            
        history['val_loss'].append(avg_val_loss)
        
        # Actualizamos el scheduler basado en la pérdida de validación
        scheduler.step(avg_val_loss)
        
        # Comprobación de Early Stopping basada en validación
        early_stopping(avg_val_loss, model)
        
        if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        if early_stopping.early_stop:
            print(f"  -> Early Stopping activado en la época {epoch+1}. Restaurando mejores pesos.")
            break
            
    # Restauramos el modelo a su mejor estado (lowest val loss)
    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
        
    return model, history


def detect_ae_anomalies(model: nn.Module, X_sequences: np.ndarray, metadata_df: pd.DataFrame, 
                     feature_names=None, device="cpu", feature_weights: dict = None) -> tuple[pd.DataFrame, float]:
    """
    Detecta anomalías en las secuencias de entrada utilizando un modelo Autoencoder entrenado.

    Calcula el error de reconstrucción (MAE) para cada secuencia y lo integra con los metadatos
    proporcionados para identificar comportamientos inusuales.

    Args:
        model: Modelo Autoencoder de PyTorch.
        X_sequences: Array de NumPy con las secuencias de entrada (N, seq_len, features).
        metadata_df: DataFrame con metadatos asociados a cada secuencia (ej. IDs, fechas).
        feature_names: Lista opcional con los nombres de las características.
        device: Dispositivo para realizar la inferencia ('cpu' o 'cuda').
        feature_weights: Diccionario opcional para ponderar variables específicas en la detección de anomalías.

    Returns:
        tuple: (DataFrame con errores por secuencia y metadatos, umbral de anomalía sugerido).
    """
    model.eval()
    model.to(device)
    
    X_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        reconstructions = model(X_tensor)
        
    # 1. ERROR GLOBAL
    # Calculamos el MAE (Mean Absolute Error) promediando la secuencia (dim=1) y las features (dim=2)
    global_errors = torch.mean(torch.abs(reconstructions - X_tensor), dim=[1, 2]).cpu().numpy()
    metadata_df[DatasetKeys.RECONSTRUCTION_ERROR] = global_errors
    
    # 2. DESGLOSE DE ERROR POR FEATURE
    # Promediamos solo a lo largo de los 12 meses (dim=1), manteniendo separadas las features
    feature_errors = torch.mean(torch.abs(reconstructions - X_tensor), dim=1).cpu().numpy()
    
    # Añadimos una columna al DataFrame por cada feature
    for i, name in enumerate(feature_names):
        metadata_df[f'error__{name}'] = feature_errors[:, i]
    
    # LÓGICA DE FLAGS Y SCORES LOCALES (POR CLÚSTER)
    # 100 significa que el error está exactamente en el umbral del percentil X
    umbral_general = np.percentile(global_errors,  AIConstants.AE_ANOMALIES_PERCENTILE)
    metadata_df[DatasetKeys.IS_GENERAL_ANOMALY] = global_errors > umbral_general
    metadata_df[DatasetKeys.AE_SCORE_GENERAL] = (global_errors / (umbral_general if umbral_general > 0 else 1e-9)) * 100
    
    if feature_weights and feature_names:
        weighted_errors = np.zeros(X_sequences.shape[0])
        total_weight = sum(feature_weights.values())
        for feature_name, weight in feature_weights.items():
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                weighted_errors += feature_errors[:, idx] * (weight / total_weight)
        
        umbral_ponderado = np.percentile(weighted_errors, AIConstants.AE_ANOMALIES_PERCENTILE)
        metadata_df[DatasetKeys.IS_WEIGHTED_ANOMALY] = weighted_errors > umbral_ponderado
        metadata_df[DatasetKeys.AE_SCORE_WEIGHTED] = (weighted_errors / (umbral_ponderado if umbral_ponderado > 0 else 1e-9)) * 100
        
        umbral_final = float(umbral_ponderado)
    else:
        metadata_df[DatasetKeys.IS_WEIGHTED_ANOMALY] = metadata_df[DatasetKeys.IS_GENERAL_ANOMALY]
        metadata_df[DatasetKeys.AE_SCORE_WEIGHTED] = metadata_df[DatasetKeys.AE_SCORE_GENERAL]
        
        umbral_final = float(umbral_general)
        
    return metadata_df, umbral_final


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