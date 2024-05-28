import torch
import torch.nn as nn

from machine_translation.seq2seq import *
from machine_translation.encoders import *
from machine_translation.decoders import *




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




# Example usage
encoder = GRUEncoder(vocabulary_size=10000, embedding_dim=64, n_hiddens=256, n_layers=2, dropout=0.2)
decoder = GRUWithAdditiveAttentionDecoder(vocabulary_size=10000, embedding_dim=64, n_hiddens=256, n_layers=2, dropout=0.2)

src = torch.randint(0, 10000, (32, 10))  # Example input
tgt = torch.randint(0, 10000, (32, 10))  # Example target

encoder_output, encoder_final_state = encoder(src)
outputs, attention_scores = decoder(tgt, encoder_output, encoder_final_state)

print(outputs.shape)  # Should be (batch_size, n_steps, vocabulary_size)
print(attention_scores[0].shape)  # Should be (batch_size, n_queries, n_keys)



