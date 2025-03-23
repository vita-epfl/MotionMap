"""
This file is from BeLFusion's codebase.
"""

import torch
from einops import rearrange

from model.seq2seq import Seq2Seq_Auto


class LSTM_Encoder(torch.nn.Module):
    def __init__(self, architecture):
        
        super(LSTM_Encoder, self).__init__()
        self.model = Seq2Seq_Auto(**architecture).cuda()


    def forward(self, x_og):
        
        # Taken from Seq2Seq_Auto code
        tf = 'b s p l f  -> s (b p) (l f)' # TODO here we need to apply 'decode' FOR EACH PARTICIPANT
        x = rearrange(x_og, tf)
        h_x = self.model._encode(x[1:])

        return h_x