from typing import Tuple, List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from .utils import masked_softmax


class BaseAttention(ABC, nn.Module):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        pass

    @abstractmethod
    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        valid_query_dims: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ScaledDotProductAttention(BaseAttention):

    def __init__(self, dropout: float):
        super().__init__()
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
            - (torch.Tensor) The output tensor of shape (batch_size, n_queries, value_dim)
            - (torch.Tensor) The attention weights tensor of shape (batch_size, n_queries, n_keys)
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
        attention_scores: torch.Tensor = masked_softmax(attention_scores=scores, valid_query_dims=valid_query_dims)
        assert attention_scores.shape == (batch_size, n_queries, n_keys)
        output: torch.Tensor = torch.bmm(input=self.dropout_layer(input=attention_scores), mat2=values)
        value_dim: int = values.shape[2]
        assert output.shape == (batch_size, n_queries, value_dim)
        return output, attention_scores


class AdditiveAttention(BaseAttention):

    def __init__(
        self, 
        query_dim: int,
        key_dim: int,
        value_dim: int,
        n_hiddens: int, 
        dropout: float
    ) -> None:

        super().__init__()
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
        valid_query_dims: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Arguments:
            - queries (torch.Tensor): The query tensor of shape (batch_size, n_queries, query_dim)
            - keys (torch.Tensor): The key tensor of shape (batch_size, n_keys, key_dim)
            - values (torch.Tensor): The value tensor of shape (batch_size, n_values, value_dim)
            - valid_query_dims (torch.Tensor): The valid length tensor of shape (batch_size,) or (batch_size, n_queries)
        
        Return:
            - (torch.Tensor) The output tensor of shape (batch_size, n_queries, value_dim)
            - (torch.Tensor) The attention weights tensor of shape (batch_size, n_queries, n_keys)
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
        attention_scores: torch.Tensor = masked_softmax(attention_scores=scores, valid_query_dims=valid_query_dims)
        assert attention_scores.shape == (batch_size, n_queries, n_keys)
        output: torch.Tensor = torch.bmm(input=self.dropout_layer(attention_scores), mat2=values)
        assert output.shape == (batch_size, n_queries, self.value_dim)
        return output, attention_scores


class EfficientMultiHeadAttention(BaseAttention):

    def __init__(
        self, 
        model_dim: int, 
        n_heads: int, 
        bias: bool, 
        dropout: float, 
    ):
        
        super().__init__()
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.dropout: float = dropout

        if model_dim % n_heads != 0:
            raise ValueError('n_hiddens must be divisible by n_heads')
        else:
            self.dim_per_head: int = model_dim // n_heads

        self.attention = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(in_features=model_dim, out_features=model_dim, bias=bias)
        self.W_k = nn.Linear(in_features=model_dim, out_features=model_dim, bias=bias)
        self.W_v = nn.Linear(in_features=model_dim, out_features=model_dim, bias=bias)
        self.W_o = nn.Linear(in_features=model_dim, out_features=model_dim, bias=bias)

    def forward(
        self,
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        valid_query_dims: torch.Tensor,
    ):
        """
        Arguments:
            - queries (torch.Tensor): The query tensor of shape (batch_size, n_queries, query_dim)
            - keys (torch.Tensor): The key tensor of shape (batch_size, n_keys, key_dim)
            - values (torch.Tensor): The value tensor of shape (batch_size, n_values, value_dim)
            - valid_query_dims (torch.Tensor): The valid length tensor of shape (batch_size,) or (batch_size, n_queries)
        
        Return:
            - (torch.Tensor) The output tensor of shape (batch_size, n_queries, value_dim)
            - (torch.Tensor) The scaled dot product attention weights tensor of shape (batch_size, n_heads, n_queries, n_keys)

        """
        batch_size: int = queries.shape[0]
        
        n_queries: int = queries.shape[1]
        query_dim: int = queries.shape[2]
        n_keys: int = keys.shape[1]
        key_dim: int = keys.shape[2]
        n_values: int = values.shape[1]
        value_dim: int = values.shape[2]

        # Check conditions
        assert n_keys == n_values
        assert query_dim == key_dim == value_dim == self.model_dim

        # Projections
        queries: torch.Tensor = self.W_q(queries)
        keys: torch.Tensor = self.W_k(keys)
        values: torch.Tesnor = self.W_v(values)

        assert queries.shape == (batch_size, n_queries, query_dim)
        assert keys.shape == (batch_size, n_keys, key_dim)
        assert values.shape == (batch_size, n_values, value_dim)

        # Break down to heads
        queries: torch.Tensor = self._break_to_heads(queries)
        keys: torch.Tensor = self._break_to_heads(keys)
        values: torch.Tensor = self._break_to_heads(values)

        # Check shapes:
        assert queries.shape == (batch_size * self.n_heads, n_queries, self.dim_per_head)
        assert keys.shape == (batch_size * self.n_heads, n_keys, self.dim_per_head)
        assert values.shape == (batch_size * self.n_heads, n_keys, self.dim_per_head)

        # Pass through scaled dot product attention
        if valid_query_dims is not None:
            valid_query_dims = torch.repeat_interleave(input=valid_query_dims, repeats=self.n_heads, dim=0)

        output: torch.Tensor
        attention_scores: torch.Tensor
        output, attention_scores = self.attention(
            queries=queries, 
            keys=keys, 
            values=values, 
            valid_query_dims=valid_query_dims, 
        )
        assert output.shape == (batch_size * self.n_heads, n_queries, self.dim_per_head)
        assert attention_scores.shape == (batch_size * self.n_heads, n_queries, n_keys)
        attention_scores: torch.Tensor = attention_scores.reshape(
            batch_size, self.n_heads, attention_scores.shape[1], attention_scores.shape[2]
        )

        # Merge from heads
        output_concat = self._merge_from_heads(output)
        assert output_concat.shape == (batch_size, n_queries, self.model_dim)
        output_concat: torch.Tensor = self.W_o(output_concat)
        assert output_concat.shape == (batch_size, n_queries, self.model_dim)
        return output_concat, attention_scores

    def _break_to_heads(self, input: torch.Tensor) -> torch.Tensor:
        """
        Transposition for parallel computation of multiple attention heads.

        Arguments:
            - input (torch.Tensor): input of shape (batch_size, n, model_dim)
        Return:
            - (torch.Tensor) of shape (batch_size * n_heads, n, dim_per_head)
        """
        # Check shape
        assert input.shape[2] == self.model_dim

        # Transformations
        x: torch.Tensor = input.reshape(input.shape[0], input.shape[1], self.n_heads, self.dim_per_head)
        x: torch.Tensor = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])
        
    def _merge_from_heads(self, input: torch.Tensor) -> torch.Tensor:
        """
        Reverse the operation of transpose_qkv.

        Arguments:
            - input (torch.Tensor): input of shape (batch_size * n_heads, n, dim_per_head)
        Return:
            - (torch.Tensor) of shape (batch_size, n, dim)
        """
        # Check shape
        assert input.shape[2] == self.dim_per_head

        # Transformations
        x: torch.Tensor = input.reshape(-1, self.n_heads, input.shape[1], input.shape[2])
        x: torch.Tensor = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)


class MultiHeadAttention(BaseAttention):

    def __init__(
        self, 
        model_dim: int, 
        n_heads: int, 
        bias: bool, 
        dropout: float, 
    ):
        
        super().__init__()
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.dropout: float = dropout


        if model_dim % n_heads != 0:
            raise ValueError('n_hiddens must be divisible by n_heads')
        else:
            self.dim_per_head: int = model_dim // n_heads

        self.attention = ScaledDotProductAttention(dropout)
        self.W_q = nn.ModuleList([
            nn.Linear(in_features=model_dim, out_features=self.dim_per_head, bias=bias)
            for _ in range(n_heads)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(in_features=model_dim, out_features=self.dim_per_head, bias=bias)
            for _ in range(n_heads)
        ])
        self.W_v = nn.ModuleList([
            nn.Linear(in_features=model_dim, out_features=self.dim_per_head, bias=bias)
            for _ in range(n_heads)
        ])
        self.W_o = nn.Linear(in_features=model_dim, out_features=self.model_dim, bias=bias)

    def forward(
        self,
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        valid_query_dims: torch.Tensor,
    ):
        """
        Arguments:
            - queries (torch.Tensor): The query tensor of shape (batch_size, n_queries, query_dim)
            - keys (torch.Tensor): The key tensor of shape (batch_size, n_keys, key_dim)
            - values (torch.Tensor): The value tensor of shape (batch_size, n_values, value_dim)
            - valid_query_dims (torch.Tensor): The valid length tensor of shape (batch_size,) or (batch_size, n_queries)
        
        Return:
            - (torch.Tensor) The output tensor of shape (batch_size, n_queries, value_dim)
            - (torch.Tensor) The scaled dot product attention weights tensor of shape (batch_size, n_heads, n_queries, n_keys)

        """
        batch_size: int = queries.shape[0]
        
        n_queries: int = queries.shape[1]
        query_dim: int = queries.shape[2]
        n_keys: int = keys.shape[1]
        key_dim: int = keys.shape[2]
        n_values: int = values.shape[1]
        value_dim: int = values.shape[2]

        # Check conditions
        assert n_keys == n_values
        assert query_dim == key_dim == value_dim == self.model_dim

        output_heads: List[torch.Tensor] = []  # each element tensor has shape (batch_size, n_queries, self.dim_per_head)
        attention_scores_heads: List[torch.Tensor] = []  # each elemenet tensor has shape (batch_size, n_queries, n_keys)
        for i in range(self.n_heads):
            
            # Projection for each head
            head_queries: torch.Tensor = self.W_q[i](queries)
            head_keys: torch.Tensor = self.W_k[i](keys)
            head_values: torch.Tensor = self.W_v[i](values)

            assert head_queries.shape == (batch_size, n_queries, self.dim_per_head)
            assert head_keys.shape == (batch_size, n_keys, self.dim_per_head)
            assert head_values.shape == (batch_size, n_values, self.dim_per_head)

            # Pass through scaled dot product attention
            output: torch.Tensor
            attention_scores: torch.Tensor
            output, attention_scores = self.attention(
                queries=head_queries, 
                keys=head_keys, 
                values=head_values, 
                valid_query_dims=valid_query_dims, 
            )
            assert output.shape == (batch_size, n_queries, self.dim_per_head)
            assert attention_scores.shape == (batch_size, n_queries, n_keys)
            output_heads.append(output)
            attention_scores_heads.append(attention_scores)

        attention_scores: torch.Tensor = torch.stack(tensors=attention_scores_heads, dim=1)
        assert attention_scores.shape == (batch_size, self.n_heads, n_queries, n_keys)

        output_concat: torch.Tensor = torch.cat(tensors=output_heads, dim=2)
        assert output_concat.shape == (batch_size, n_queries, self.model_dim)
        output_concat: torch.Tensor = self.W_o(output_concat)
        assert output_concat.shape == (batch_size, n_queries, self.model_dim)

        return output_concat, attention_scores










if __name__ == '__main__':

    queries = torch.normal(mean=0, std=1, size=(2, 1, 20))
    keys = torch.normal(mean=0, std=1, size=(2, 10, 2))
    values = torch.normal(mean=0, std=1, size=(2, 10, 4))
    valid_dims = torch.tensor(data=[2, 6])

    attention = AdditiveAttention(n_hiddens=8, dropout=0.1)
    attention.eval()
    y = attention(queries=queries, keys=keys, values=values, valid_query_dims=valid_dims)
    print(y)



