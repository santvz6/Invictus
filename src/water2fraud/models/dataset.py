import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class WaterSeriesDataset(Dataset):
    """
    Clase Dataset personalizada para envolver secuencias de series temporales en PyTorch.
    
    Facilita la carga de datos en lotes (batches) durante el entrenamiento del Autoencoder,
    devolviendo la misma secuencia tanto como entrada como objetivo (input = output).
    """

    def __init__(self, sequences: np.ndarray) -> None:
        """
        Inicializa el dataset convirtiendo los arrays de NumPy en tensores de PyTorch.

        Args:
            sequences (np.ndarray): Array con las secuencias temporales de forma
                                    (num_muestras, seq_length, num_features).
        """
        # Convertimos a tensores float32
        self.X = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Devuelve el número total de muestras en el dataset.

        Returns:
            int: Cantidad de secuencias temporales.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene una muestra específica del dataset por su índice.

        Args:
            idx (int): Índice de la muestra a recuperar.

        Returns:
            tuple: Par de tensores (entrada, objetivo) que en un autoencoder son idénticos.
        """
        # Autoencoder: Input = Output
        return self.X[idx], self.X[idx]


def get_dataloader(sequences: np.ndarray, batch_size=32, shuffle=True) -> DataLoader:
    """
    Genera un DataLoader de PyTorch a partir de las secuencias proporcionadas.

    Args:
        sequences (np.ndarray): Secuencias temporales a cargar.
        batch_size (int, optional): Tamaño del lote para el entrenamiento. Por defecto es 32.
        shuffle (bool, optional): Si es True, mezcla las muestras en cada época. Por defecto es True.

    Returns:
        DataLoader: Objeto iterador para el entrenamiento o evaluación del modelo.
    """
    dataset = WaterSeriesDataset(sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)