import torch
import torch.nn as nn
from typing import Optional

class BasePredictiveModel(nn.Module):
    """Base class for all predictive maintenance models."""
    def __init__(self, name: str):
        super(BasePredictiveModel, self).__init__()
        self.name = name

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

class LSTMPredictor(BasePredictiveModel):
    """
    LSTM-based model for time-series failure prediction.
    Suitable for capturing temporal dependencies in sensor data.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 1):
        super(LSTMPredictor, self).__init__(name="LSTM_Predictor")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last time step's output
        last_time_step = lstm_out[:, -1, :]
        
        out = self.fc(last_time_step)
        return out, hidden

class AnomalyAutoencoder(BasePredictiveModel):
    """
    Autoencoder for unsupervised anomaly detection.
    Reconstruction error is used as an anomaly score.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super(AnomalyAutoencoder, self).__init__(name="Anomaly_Autoencoder")
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
