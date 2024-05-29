import torch
import torch.nn as nn

from machine_translation.seq2seq import *
from machine_translation.encoders import *
from machine_translation.decoders import *
from attention import *




# vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
# batch_size, num_steps = 4, 7

# encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
# decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens, num_layers)
# X = torch.zeros((batch_size, num_steps), dtype=torch.long)
# state0 = decoder.init_state(encoder(X), None)
# output, state = decoder(X, state0)
# print(output)
# print(state)

# decoder = GRUWithAdditiveAttentionDecoder(vocabulary_size=vocab_size, embedding_dim=embed_size, n_hiddens=num_hiddens, n_layers=1, dropout=0)
# X = torch.zeros((batch_size, num_steps), dtype=torch.long)
# output, state = decoder(x=X, encoder_output=state0[0], encoder_final_state=state0[1], valid_query_dim=None)
# print(output)
# print(state)




num_hiddens, num_heads = 100, 5
batch_size, num_queries, num_kvpairs = 2, 4, 6

valid_lens = torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

attention1 = EfficientMultiHeadAttention(model_dim=num_hiddens, n_heads=num_heads, dropout=0.5, bias=False)
r1 = attention1(X, Y, Y, valid_lens)

attention2 = MultiHeadAttention(model_dim=num_hiddens, n_heads=num_heads, dropout=0.5, bias=False)
r2 = attention2(X, Y, Y, valid_lens)


