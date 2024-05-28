from typing import Tuple, List, Optional

import torch
import torch.nn as nn

from attention import AdditiveAttention


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

        self.encoder_projection = nn.Linear(n_hiddens * 2, n_hiddens)
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size, 
            embedding_dim=embedding_dim,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim + n_hiddens, 
            hidden_size=n_hiddens, 
            num_layers=n_layers, 
            dropout=dropout, 
        )
        self.dense = nn.Linear(in_features=self.n_hiddens, out_features=vocabulary_size)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        encoder_final_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Arguments:
            - x (torch.Tensor): decoder input tensor of shape (batch_size, embedding_dim)
            - encoder_output (torch.Tensor): hidden state of the last layer of the encoder at all timesteps, 
            shape (n_steps, batch_size, n_hiddens * 2)
            - encoder_final_state (torch.Tensor): hidden state of the encoder at final timestep of all layers,
            shape (n_layers * 2, batch_size, n_hiddens)
        Returns:
            - (torch.Tensor) the outputs of the decoder, shape (batch_size, n_steps, vocabulary_size)
            - (torch.Tensor) the hidden state of the encoder at final timestep of all layers, 
            shape (n_layers, batch_size, n_hiddens)
        """

        batch_size, n_steps = x.shape
        # Check shapes
        assert encoder_output.shape == (n_steps + 1, batch_size, self.n_hiddens * 2)
        assert encoder_final_state.shape == (self.n_layers * 2, batch_size, self.n_hiddens)

        # Encoder Projection
        projected_encoder_output: torch.Tensor = self.encoder_projection(input=encoder_output)
        assert projected_encoder_output.shape == (n_steps + 1, batch_size, self.n_hiddens)

        # Handle bidirectional output from encoder
        merged_encoder_final_state: torch.Tensor = encoder_final_state.reshape(self.n_layers, 2, batch_size, self.n_hiddens)
        merged_encoder_final_state: torch.Tensor = merged_encoder_final_state.sum(dim=1)
        assert merged_encoder_final_state.shape == (self.n_layers, batch_size, self.n_hiddens)

        # It requires that the RNN encoder and the RNN decoder have the same number of layers and hidden units
        embeddings: torch.Tensor = self.embedding(input=x.t().contiguous().type(dtype=torch.int64))
        assert embeddings.shape == (n_steps, batch_size, self.embedding_dim)
        context = projected_encoder_output[-1]  # shape (batch_size, n_hiddens)
        context = context.repeat(n_steps, 1, 1) # broadcast to (n_steps, batch_size, n_hiddens)
        assert context.shape == (n_steps, batch_size, self.n_hiddens)

        # The context variable is concatenated with the decoder input at all the time steps
        embeddings_and_context = torch.cat(tensors=(embeddings, context), dim=2) # shape (n_steps, batch_size, embedding_dim + n_hiddens)

        # Assign a predicted probability to each possible token
        output: torch.Tensor        # shape (n_steps, batch_size, n_hiddens)
        hidden_state: torch.Tensor  # shape (n_layers, batch_size, n_hiddens)
        output, hidden_state = self.gru(input=embeddings_and_context, hx=merged_encoder_final_state)
        assert output.shape == (n_steps, batch_size, self.n_hiddens)
        assert hidden_state.shape == (self.n_layers, batch_size, self.n_hiddens)
        output = self.dense(output).permute(1, 0, 2)
        assert output.shape == (batch_size, n_steps, self.vocabulary_size)
        return output, hidden_state
    

class GRUWithAdditiveAttentionDecoder(nn.Module):

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
        
        self.encoder_projection = nn.Linear(n_hiddens * 2, n_hiddens)
        self.attention = AdditiveAttention(
            query_dim=n_hiddens,
            key_dim=n_hiddens,
            value_dim=n_hiddens,
            n_hiddens=n_hiddens, 
            dropout=0.2,
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size, 
            embedding_dim=embedding_dim,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim + n_hiddens, 
            hidden_size=n_hiddens, 
            num_layers=n_layers, 
            dropout=dropout, 
        )
        self.dense = nn.Linear(in_features=self.n_hiddens, out_features=vocabulary_size)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        encoder_final_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Arguments:
            - x (torch.Tensor): decoder input tensor of shape (batch_size, embedding_dim)
            - encoder_output (torch.Tensor): hidden state of the last layer of the encoder at all timesteps, 
            shape (n_steps, batch_size, n_hiddens * 2)
            - encoder_final_state (torch.Tensor): hidden state of the encoder at final timestep of all layers,
            shape (n_layers * 2, batch_size, n_hiddens)
        Returns:
            - (torch.Tensor) the outputs of the decoder, shape (batch_size, n_steps, vocabulary_size)
            - (torch.Tensor) attention scores tensor of shape (batch_size, n_steps, n_steps * 2)
        """

        batch_size, n_steps = x.shape
        # Check shapes
        assert encoder_output.shape == (n_steps + 1, batch_size, self.n_hiddens * 2)
        assert encoder_final_state.shape == (self.n_layers * 2, batch_size, self.n_hiddens)

        # Encoder Projection
        projected_encoder_output: torch.Tensor = self.encoder_projection(input=encoder_output.permute(1, 0, 2))
        assert projected_encoder_output.shape == (batch_size, n_steps + 1, self.n_hiddens)

        # Handle bidirectional output from encoder
        merged_encoder_final_state: torch.Tensor = encoder_final_state.reshape(self.n_layers, 2, batch_size, self.n_hiddens)
        merged_encoder_final_state: torch.Tensor = merged_encoder_final_state.sum(dim=1)
        assert merged_encoder_final_state.shape == (self.n_layers, batch_size, self.n_hiddens)

        embeddings: torch.Tensor = self.embedding(input=x.t().contiguous().type(dtype=torch.int64))
        assert embeddings.shape == (n_steps, batch_size, self.embedding_dim)

        # The hidden state of the encoder at all layers at the final time step serves to initialize the hidden state of the decoder
        hidden_state: torch.Tensor = merged_encoder_final_state
        
        # Decode at each time step
        outputs: List[torch.Tensor] = []
        attention_scores: List[torch.Tensor] = []
        for timestep, embedding in enumerate(embeddings):
            # At each decoding time step, the hidden state of the final layer of the decoder, obtained at the previous 
            # time step, is used as the query of the attention mechanism
            query: torch.Tensor = hidden_state[-1].unsqueeze(dim=1)
            assert query.shape == (batch_size, 1, self.n_hiddens)   # n_queries == 1, query_dim == self.n_hiddens

            # Combine encoder outputs and decoder outputs for attention (concat along n_steps axis)
            combined_output: torch.Tensor = torch.cat(tensors=[projected_encoder_output] + outputs, dim=1)
            assert combined_output.shape == (batch_size, n_steps + 1 + len(outputs), self.n_hiddens)

            # The hidden states of the last layer of the encoder at all time steps combined with the hidden states generated by
            # the decoder are used as keys and values for attention
            context: torch.Tensor           # (batch_size, n_queries, value_dim)
            attention_score: torch.Tensor   # (batch_size, n_queries, n_keys)
            context, attention_score = self.attention(
                queries=query,
                keys=combined_output,
                values=combined_output,
                valid_query_dim=None,   # No mask
            )
            assert context.shape == (batch_size, 1, self.n_hiddens) # key_dim == value_dim == self.n_hiddens
            assert attention_score.shape == (batch_size, 1, combined_output.shape[1]) # n_keys == number of steps

            feature: torch.Tensor = torch.cat(
                tensors=[context, embedding.unsqueeze(dim=1)],
                dim=2,
            )
            assert feature.shape == (batch_size, 1, self.n_hiddens + self.embedding_dim)

            # context c_t is then used to generate the state s_t and to generate a new token
            output: torch.Tensor; hidden_state: torch.Tensor
            output, hidden_state = self.gru(input=feature.permute(1, 0, 2), hx=hidden_state)
            assert output.shape == (1, batch_size, self.n_hiddens)
            assert hidden_state.shape == (self.n_layers, batch_size, self.n_hiddens)

            outputs.append(output.permute(1, 0, 2))
            attention_scores.append(attention_score)
        
        outputs: torch.Tensor = torch.cat(outputs, dim=1)
        assert outputs.shape == (batch_size, n_steps, self.n_hiddens)
        outputs: torch.Tensor = self.dense(input=outputs)
        assert outputs.shape == (batch_size, n_steps, self.vocabulary_size)

        # Right pad `attention_scores` to the maximum length with value 0
        max_length: int = 2 * n_steps
        padded_attention_scores = []
        for attention_score in attention_scores:
            pad_size: int = max_length - attention_score.shape[2]
            padded_attention_score: torch.Tensor = nn.functional.pad(
                input=attention_score, 
                pad=(0, pad_size),
                mode='constant',
                value=0,
            )
            padded_attention_scores.append(padded_attention_score)

        attention_scores: torch.Tensor = torch.cat(padded_attention_scores, dim=1)
        assert attention_scores.shape == (batch_size, n_steps, n_steps * 2)

        return outputs, attention_scores




