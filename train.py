from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn import CrossEntropyLoss

from machine_translation.encoders import GRUEncoder
from machine_translation.decoders import GRUDecoder
from machine_translation.seq2seq import Seq2Seq
from machine_translation.datasets import EnglishFrenchDataset

dataset: Dataset = EnglishFrenchDataset(
    txt_path="./data/fra-eng/fra.txt", 
    num_steps=10, 
    device=torch.device('cpu'),
)
train_set: Subset; val_set: Subset; test_set: Subset
train_set, val_set, test_set = random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1])
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)

encoder = GRUEncoder(
    vocabulary_size=len(dataset.source_vocab), 
    embedding_dim=256, 
    n_hiddens=256, 
    n_layers=2, 
    dropout=0.2,
)
decoder = GRUDecoder(
    vocabulary_size=len(dataset.target_vocab), 
    embedding_dim=256, 
    n_hiddens=256, 
    n_layers=2, 
    dropout=0.2,
)
model = Seq2Seq(encoder=encoder, decoder=decoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = CrossEntropyLoss()

max_epochs: int = 30

for epoch in range(max_epochs):
    batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    for i, batch in enumerate(train_dataloader):
        source_tensor, source_valid_len = batch['source'] # source_tensor (batch_size, n_steps); source_valid_len (batch_size,)
        target_tensor, target_valid_len = batch['target'] # target_tensor (batch_size, n_steps); target_valid_len (batch_size,)
        optimizer.zero_grad()
        output, hidden_state = model(enc_X=source_tensor, dec_X=target_tensor)
        loss = criterion(input=output.reshape(-1, output.shape[-1]), target=target_tensor.reshape(-1))
        loss.backward()
        optimizer.step()





