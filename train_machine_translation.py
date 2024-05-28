from typing import Dict, Tuple, Optional
import argparse

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn as nn

from machine_translation.encoders import GRUEncoder
from machine_translation.decoders import GRUDecoder, GRUWithAdditiveAttentionDecoder
from machine_translation.seq2seq import Seq2Seq
from machine_translation.datasets import EnglishFrenchDataset
from machine_translation.training import Trainer


device: torch.device = torch.device('cpu')

dataset: Dataset = EnglishFrenchDataset(txt_path="./data/fra-eng/fra.txt", num_steps=10, device=device)

encoder = GRUEncoder(
    vocabulary_size=len(dataset.source_vocab), 
    embedding_dim=128, 
    n_hiddens=128, 
    n_layers=5, 
    dropout=0.2,
)
# decoder = GRUDecoder(
#     vocabulary_size=len(dataset.target_vocab), 
#     embedding_dim=128, 
#     n_hiddens=128, 
#     n_layers=5, 
#     dropout=0.2,
# )

decoder = GRUWithAdditiveAttentionDecoder(
    vocabulary_size=len(dataset.target_vocab), 
    embedding_dim=128, 
    n_hiddens=128, 
    n_layers=5, 
    dropout=0.2,
)

model = Seq2Seq(encoder=encoder, decoder=decoder, device=device)

trainer = Trainer(
    model=model, dataset=dataset, split_ratios=(0.8, 0.1, 0.1), 
    train_batch_size=16, val_batch_size=16, test_batch_size=16, 
    learning_rate=0.005,
)
trainer.train(n_epochs=10, patience=3, tolerance=0.)


