import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

from src.model_training.architectures import LSTMPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu"
):
    """
    Standard training loop for the predictive models.
    """
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    logger.info("Training complete.")

if __name__ == "__main__":
    # Example usage with dummy data
    input_dim = 10
    seq_len = 30
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)
    
    # Generate random dummy data
    dummy_x = torch.randn(100, seq_len, input_dim)
    dummy_y = torch.randint(0, 2, (100, 1)).float()
    
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    train_model(model, loader, num_epochs=5)
