import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim=19, embed_dim=96, conv_channels=128, hidden_dim=160, num_classes=8, dropout=0.25):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels), nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.input_norm(x)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        context, weights = self.attention(x)
        logits = self.classifier(context)
        if return_attention:
            return logits, weights
        return logits
