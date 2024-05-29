from typing import Optional

import torch
import torch.nn as nn


def masked_softmax(attention_scores: torch.Tensor, valid_query_dims: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Perform masked softmax operation

    Arguments:
        - attention_scores (torch.Tensor): input tensor of shape (batch_size, n_queries, n_keys)
        - valid_query_dims (torch.Tensor): the valid length of each sample, 
        shape (batch_size,) or (batch_size, n_queries)
    
    Returns:
        - (torch.Tensor) The output tensor of shape (batch_size, n_queries, n_keys)
    """

    batch_size: int = attention_scores.shape[0]
    n_queries: int = attention_scores.shape[1]
    n_keys: int = attention_scores.shape[2]

    if valid_query_dims is None:    # no masks
        return nn.functional.softmax(input=attention_scores, dim=2)

    if valid_query_dims.dim() == 1:
        # every samples have the same number of queries of the same valid dim
        # (batch_size,) -> (batch_size * n_queries,)
        valid_query_dims: torch.Tensor = torch.repeat_interleave(
            input=valid_query_dims, repeats=n_queries
        )
    elif valid_query_dims.dim() == 2:
        # every samples have the same number of queries of different valid dims
        # (batch_size, n_queries) -> (batch_size * n_queries,)
        valid_query_dims: torch.Tensor = valid_query_dims.reshape(-1)
    else:
        raise ValueError('Invalid shape of `valid_query_dims`')

    mask: torch.Tensor = (
        torch.arange(n_keys, dtype=torch.float32, device=attention_scores.device)[None, :] 
        < valid_query_dims[:, None]
    )
    assert mask.shape == (batch_size * n_queries, n_keys)
    attention_scores.reshape(-1, attention_scores.shape[2])[~mask] = -1e6
    return nn.functional.softmax(input=attention_scores, dim=2)  # for each query, compute softmax over keys


