import torch
import torch.nn as nn


class StockLSTM(nn.Module):

  def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.2):
    super().__init__()

    self.lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )

    self.fc = nn.Sequential(
        nn.Linear(hidden_size, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

  def forward(self, x):
    # If single time-step [batch_size, features], convert to [batch_size, 1, features]
    if x.dim() == 2:
      x = x.unsqueeze(1)

    out, _ = self.lstm(x)

    # Take the output state of the last time step in the sequence
    out = out[:, -1, :]

    return self.fc(out)