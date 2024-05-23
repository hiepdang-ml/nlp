from typing import Tuple, List

import torch
import torch.nn as nn

from .encoders import GRUEncoder
from .decoders import GRUDecoder


class Seq2Seq(nn.Module):

    def __init__(self, encoder: GRUEncoder, decoder: GRUDecoder, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.encoder: GRUEncoder = encoder.to(device=device)
        self.decoder: GRUDecoder = decoder.to(device=device)
        self.device = device

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass with teacher forcing """
        enc_output: torch.Tensor
        enc_final_state: torch.Tensor
        enc_output, enc_final_state = self.encoder(x=enc_x)
        return self.decoder(x=dec_x, encoder_output=enc_output, encoder_final_state=enc_final_state)

    def predict(self, source_token_ids: List[int], bos_id: int):
        """ Forward pass without teacher forcing """
        self.eval()
        with torch.no_grad():
            enc_output: torch.Tensor
            enc_final_state: torch.Tensor
            enc_output, enc_final_state = self.encoder(
                x=torch.tensor(data=source_token_ids, device=self.device).unsqueeze(dim=0)
            )
            target_token_ids: List[int] = [bos_id]
            for _ in range(len(source_token_ids)):
                dec_output: torch.Tensor = self.decoder(
                    x=torch.tensor(data=target_token_ids, device=self.device).unsqueeze(dim=0), 
                    encoder_output=enc_output, 
                    encoder_final_state=enc_final_state, 
                )[0]
                predicted_next_token_id = dec_output.argmax(dim=2)[0, -1].item()
                target_token_ids.append(predicted_next_token_id)

        return target_token_ids

    @staticmethod
    def from_pretrained(checkpoint_path: str, device: torch.device = torch.device('cpu')):
        model: nn.Module = torch.load(checkpoint_path, map_location=device)
        return model



