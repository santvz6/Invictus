import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_autoencoder(model: nn.Module, dataloader: DataLoader, epochs=50, lr=1e-3, device="cpu") -> nn.Module:
    """
    Bucle de entrenamiento estandarizado para el Autoencoder LSTM.
    
    Minimiza el Error Absoluto Medio (MAE / L1Loss) entre la secuencia original
    y la reconstrucción generada por el modelo.

    Args:
        model (nn.Module): Instancia de la red neuronal (LSTMAutoencoder) a entrenar.
        dataloader (DataLoader): Iterador de PyTorch con los datos de entrenamiento.
        epochs (int, optional): Número de pasadas completas por el dataset. Por defecto es 50.
        lr (float, optional): Tasa de aprendizaje (Learning Rate) para el optimizador Adam. Por defecto es 1e-3.
        device (str, optional): Dispositivo de cómputo ('cpu' o 'cuda'). Por defecto es 'cpu'.

    Returns:
        nn.Module: El modelo entrenado con los pesos actualizados.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() # MAE suele ser mejor que MSE para anomalías, es menos sensible a picos extremos
    
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
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    return model


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