from typing import Tuple

import torch
import torch.nn as nn

from .encoders import GRUEncoder
from .decoders import GRUDecoder


class Seq2Seq(nn.Module):

    def __init__(self, encoder: GRUEncoder, decoder: GRUDecoder):
        super().__init__()
        self.encoder: GRUEncoder = encoder
        self.decoder: GRUDecoder = decoder

    def forward(self, enc_X: torch.Tensor, dec_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_output, enc_final_state = self.encoder(x=enc_X)
        return self.decoder(x=dec_X, encoder_output=enc_output, encoder_final_state=enc_final_state)



