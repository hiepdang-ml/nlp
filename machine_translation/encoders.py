from typing import Tuple

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):

    def __init__(
        self, 
        vocabulary_size: int,
        embedding_dim: int,
        n_hiddens: int,
        n_layers: int,
        dropout: int,
    ):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_hiddens: int = n_hiddens
        self.n_layers: int = n_layers
        self.dropout: int = dropout
        self.vocabulary_size: int = vocabulary_size

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=n_hiddens, num_layers=n_layers, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_steps = x.shape
        embeddings: torch.Tensor = self.embedding(input=x.t().contiguous().type(dtype=torch.int64)) # shape (n_steps, batch_size, embedding_dim)
        assert embeddings.shape == (n_steps, batch_size, self.embedding_dim)
        output: torch.Tensor
        final_state: torch.Tensor
        output, final_state = self.gru(input=embeddings)
        assert output.shape == (n_steps, batch_size, self.n_hiddens)
        assert final_state.shape == (self.n_layers, batch_size, self.n_hiddens)
        return output, final_state




