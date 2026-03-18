import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    Autoencoder basado en redes LSTM para la compresión y reconstrucción de series temporales multivariantes.
        
    Esta arquitectura codifica una secuencia temporal en un vector latente de menor dimensión
    y posteriormente la decodifica para reconstruir la entrada original. Es ideal para capturar
    patrones estacionales y detectar anomalías mediante el error de reconstrucción.
    """

    def __init__(self, num_features: int, hidden_dim:int, latent_dim:int, seq_len:int) -> None:
        """
        Inicializa la arquitectura del LSTMAutoencoder.

        Args:
            num_features (int): Número de variables (características) por cada paso temporal.
            hidden_dim (int, optional): Dimensión de la primera capa oculta LSTM. Por defecto es 64.
            latent_dim (int, optional): Dimensión del espacio latente (cuello de botella). Por defecto es 16.
            seq_len (int, optional): Longitud de la secuencia temporal (ej. 12 meses). Por defecto es 12.
        """
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        # ENCODER
        self.encoder_lstm1 = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.encoder_dropout = nn.Dropout(0.2)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)

        # DECODER
        self.decoder_lstm1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder_dropout = nn.Dropout(0.2)
        self.decoder_lstm2 = nn.LSTM(hidden_dim, num_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define el flujo hacia adelante (forward pass) de la red.

        Args:
            x (torch.Tensor): Tensor de entrada con forma (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Tensor reconstruido con la misma forma que la entrada.
        """
        
        # --- Encode ---
        x, _ = self.encoder_lstm1(x)
        x = self.encoder_dropout(x)
        _, (hidden, cell) = self.encoder_lstm2(x)
        
        # hidden shape: (1, batch_size, latent_dim) -> squeeze: (batch_size, latent_dim)
        latent = hidden[-1]

        # --- Decode ---
        # Repetimos el vector latente a lo largo de los meses
        # shape: (batch_size, seq_len, latent_dim)
        x_decoded = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        x_decoded, _ = self.decoder_lstm1(x_decoded)
        x_decoded = self.decoder_dropout(x_decoded)
        reconstruction, _ = self.decoder_lstm2(x_decoded)
        
        return reconstruction