from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_categories: int, dropout: float):
        super().__init__()
        self.purchase_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.category_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_categories),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        purchase_logits = self.purchase_head(features).squeeze(-1)
        category_logits = self.category_head(features)
        return purchase_logits, category_logits


class StaticFeatureEncoder(nn.Module):
    def __init__(self, static_input_size: int, output_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(static_input_size, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, static_x: torch.Tensor | None) -> torch.Tensor:
        if static_x is None:
            raise ValueError("Static features are required for this model.")
        return self.encoder(static_x)


class LSTMPredictor(nn.Module):
    def __init__(
        self,
        seq_input_size: int = 5,
        static_input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_categories: int = 21,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_encoder = StaticFeatureEncoder(static_input_size, hidden_size // 2, dropout)
        self.head = ClassificationHead(hidden_size + hidden_size // 2, hidden_size, num_categories, dropout)

    def forward(
        self, seq_x: torch.Tensor, static_x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(seq_x)
        seq_features = encoded[:, -1, :]
        static_features = self.static_encoder(static_x)
        features = torch.cat([seq_features, static_features], dim=1)
        return self.head(features)


class GRUPredictor(nn.Module):
    def __init__(
        self,
        seq_input_size: int = 5,
        static_input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_categories: int = 21,
    ):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_encoder = StaticFeatureEncoder(static_input_size, hidden_size // 2, dropout)
        self.head = ClassificationHead(hidden_size + hidden_size // 2, hidden_size, num_categories, dropout)

    def forward(
        self, seq_x: torch.Tensor, static_x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(seq_x)
        seq_features = encoded[:, -1, :]
        static_features = self.static_encoder(static_x)
        features = torch.cat([seq_features, static_features], dim=1)
        return self.head(features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderPredictor(nn.Module):
    def __init__(
        self,
        seq_input_size: int = 5,
        static_input_size: int = 11,
        seq_length: int = 10,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_categories: int = 21,
    ):
        super().__init__()
        self.input_projection = nn.Linear(seq_input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.static_encoder = StaticFeatureEncoder(static_input_size, hidden_size // 2, dropout)
        self.head = ClassificationHead(hidden_size + hidden_size // 2, hidden_size, num_categories, dropout)

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_features = self.input_projection(seq_x)
        seq_features = self.position(seq_features)
        encoded = self.encoder(seq_features)
        non_padding = (seq_x.abs().sum(dim=-1) > 0).float()
        lengths = non_padding.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (encoded * non_padding.unsqueeze(-1)).sum(dim=1) / lengths
        static_features = self.static_encoder(static_x)
        features = torch.cat([pooled, static_features], dim=1)
        return self.head(features)


class MLPFusionPredictor(nn.Module):
    def __init__(
        self,
        seq_input_size: int = 5,
        seq_length: int = 10,
        static_input_size: int = 11,
        hidden_size: int = 128,
        dropout: float = 0.2,
        num_categories: int = 21,
    ):
        super().__init__()
        fused_input = seq_input_size * seq_length + static_input_size
        self.flatten = nn.Flatten()
        self.backbone = nn.Sequential(
            nn.Linear(fused_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = ClassificationHead(64, 64, num_categories, dropout)

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_features = self.flatten(seq_x)
        if static_x is None:
            raise ValueError("MLPFusionPredictor requires static features.")
        features = torch.cat([seq_features, static_x], dim=1)
        features = self.backbone(features)
        return self.head(features)


MODEL_REGISTRY = {
    "lstm": LSTMPredictor,
    "gru": GRUPredictor,
    "mlp_fusion": MLPFusionPredictor,
    "transformer": TransformerEncoderPredictor,
}
