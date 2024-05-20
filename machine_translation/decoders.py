from typing import Tuple

import torch
import torch.nn as nn


class GRUDecoder(nn.Module):

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
        self.gru = nn.GRU(input_size=embedding_dim + n_hiddens, hidden_size=n_hiddens, num_layers=n_layers, dropout=dropout)
        self.dense = nn.Linear(in_features=self.n_hiddens, out_features=vocabulary_size)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        encoder_final_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, n_steps = x.shape
        assert encoder_output.shape == (n_steps, batch_size, self.n_hiddens)
        # It requires that the RNN encoder and the RNN decoder have the same number of layers and hidden units
        assert encoder_final_state.shape == (self.n_layers, batch_size, self.n_hiddens)
        embeddings: torch.Tensor = self.embedding(input=x.t().contiguous().type(dtype=torch.int64))
        assert embeddings.shape == (n_steps, batch_size, self.embedding_dim)
        context = encoder_output[-1]    # shape (batch_size, n_hiddens)
        context = context.repeat(n_steps, 1, 1) # broadcast to (n_steps, batch_size, n_hiddens)
        assert context.shape == (n_steps, batch_size, self.n_hiddens)
        # the context variable is concatenated with the decoder input at all the time steps
        embeddings_and_context = torch.cat(tensors=(embeddings, context), dim=2) # shape (n_steps, batch_size, embedding_dim + n_hiddens)
        output: torch.Tensor    # shape (n_steps, batch_size, n_hiddens)
        hidden_state: torch.Tensor  # shape (n_layers, batch_size, n_hiddens)
        # assign a predicted probability to each possible token
        output, hidden_state = self.gru(input=embeddings_and_context, hx=encoder_final_state)
        assert output.shape == (n_steps, batch_size, self.n_hiddens)
        assert hidden_state.shape == (self.n_layers, batch_size, self.n_hiddens)
        output = self.dense(output).swapaxes(0, 1)  # shape (batch_size, n_steps, vocabulary_size)
        return output, hidden_state
    


