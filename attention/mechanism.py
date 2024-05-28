from typing import Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from .utils import masked_softmax


class BaseAttention(ABC, nn.Module):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        valid_query_dim: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ScaledDotProductAttention(BaseAttention):

    def __init__(self, dropout: float):
        self.dropout: float = dropout
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        valid_query_dims: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Arguments:
            - queries (torch.Tensor): The query tensor of shape (batch_size, n_queries, query_dim)
            - keys (torch.Tensor): The key tensor of shape (batch_size, n_keys, key_dim)
            - values (torch.Tensor): The value tensor of shape (batch_size, n_values, value_dim)
            - valid_query_dims (torch.Tensor): The valid length tensor of shape (batch_size,) or (batch_size, n_queries)
        
        Return:
            - (torch.Tensor) the output tensor of shape (batch_size, n_queries, value_dim)
            - (torch.Tensor) the attention weights tensor of shape (batch_size, n_queries, n_keys)
        """

        n_keys: int = keys.shape[1]
        n_values: int = values.shape[1]
        assert n_keys == n_values, '`n_keys` should be equal to `n_values` (to form key-value pairs)'
        
        query_dim: int = queries.shape[2]
        key_dim: int = keys.shape[2]
        assert query_dim == key_dim, '`query_dim` should be equal to `key_dim`'

        batch_size: int = queries.shape[0]
        n_queries: int = queries.shape[1]
        query_dim: int = queries.shape[2]
        scores: torch.Tensor = torch.bmm(input=queries, mat2=keys.transpose(1, 2)) / (query_dim ** .5)
        assert scores.shape == (batch_size, n_queries, n_keys)
        attention_weights: torch.Tensor = masked_softmax(attention_scores=scores, valid_query_dims=valid_query_dims)
        assert attention_weights.shape == (batch_size, n_queries, n_keys)
        output: torch.Tensor = torch.bmm(input=self.dropout_layer(input=attention_weights), mat2=values)
        value_dim: int = values.shape[2]
        assert output.shape == (batch_size, n_queries, value_dim)
        return output, attention_weights


class AdditiveAttention(BaseAttention):

    def __init__(
        self, 
        query_dim: int,
        key_dim: int,
        value_dim: int,
        n_hiddens: int, 
        dropout: float
    ) -> None:
        
        self.query_dim: int = query_dim
        self.key_dim: int = key_dim
        self.value_dim: int = value_dim
        self.n_hiddens: int = n_hiddens
        self.dropout: float = dropout
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout)
        self.W_k = nn.Linear(in_features=key_dim, out_features=n_hiddens, bias=False)
        self.W_q = nn.Linear(in_features=query_dim, out_features=n_hiddens, bias=False)
        self.w_v = nn.Linear(in_features=value_dim, out_features=1, bias=False)

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        valid_query_dim: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Arguments:
            - queries (torch.Tensor): The query tensor of shape (batch_size, n_queries, query_dim)
            - keys (torch.Tensor): The key tensor of shape (batch_size, n_keys, key_dim)
            - values (torch.Tensor): The value tensor of shape (batch_size, n_values, value_dim)
            - valid_query_dims (torch.Tensor): The valid length tensor of shape (batch_size,) or (batch_size, n_queries)
        
        Return:
            - (torch.Tensor) the output tensor of shape (batch_size, n_queries, value_dim)
            - (torch.Tensor) the attention weights tensor of shape (batch_size, n_queries, n_keys)
        """

        assert queries.shape[2] == self.query_dim
        assert keys.shape[2] == self.key_dim
        assert values.shape[2] == self.value_dim
        
        batch_size: int = queries.shape[0]
        n_queries: int = queries.shape[1]
        n_keys: int = keys.shape[1]
        n_values: int = values.shape[1]
        assert batch_size == keys.shape[0] == values.shape[0]
        assert n_keys == n_values

        queries: torch.Tensor = self.W_q(queries)
        assert queries.shape == (batch_size, n_queries, self.n_hiddens)
        keys: torch.Tensor = self.W_k(keys)
        assert keys.shape == (batch_size, n_keys, self.n_hiddens)

        features: torch.Tensor = queries[:, :, None, :] + keys[:, None, :, :]
        assert features.shape == (batch_size, n_queries, n_keys, self.n_hiddens)
        features: torch.Tensor = torch.tanh(features)
        assert features.shape == (batch_size, n_queries, n_keys, self.n_hiddens)
        
        scores: torch.Tensor = self.w_v(features).squeeze(3)
        assert scores.shape == (batch_size, n_queries, n_keys)
        attention_weight: torch.Tensor = masked_softmax(attention_scores=scores, valid_query_dims=valid_query_dim)
        assert attention_weight.shape == (batch_size, n_queries, n_keys)
        output: torch.Tensor = torch.bmm(input=self.dropout_layer(attention_weight), mat2=values)
        assert output.shape == (batch_size, n_queries, self.value_dim)
        return output, attention_weight


class MultiHeadAttention(BaseAttention):

    def __init__(self, n_hiddens: int, n_heads: int, dropout: float, bias=False):
        self.n_hiddens: int = n_hiddens
        self.num_heads: int = n_heads
        self.dropout: float = dropout

        self.attention = ScaledDotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)



if __name__ == '__main__':

    queries = torch.normal(mean=0, std=1, size=(2, 1, 20))
    keys = torch.normal(mean=0, std=1, size=(2, 10, 2))
    values = torch.normal(mean=0, std=1, size=(2, 10, 4))
    valid_dims = torch.tensor(data=[2, 6])

    attention = AdditiveAttention(n_hiddens=8, dropout=0.1)
    attention.eval()
    y = attention(queries=queries, keys=keys, values=values, valid_query_dim=valid_dims)
    print(y)



