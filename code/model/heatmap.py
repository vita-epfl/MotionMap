import torch
import math

import copy
from model.encoder import LSTM_Encoder


class HeatmapDecoder(torch.nn.Module):
    """
    A PyTorch module for decoding observation into latent representations into heatmaps.

    Attributes:
        hm_size (int): The size of the heatmap (height and width).
        layernorm (torch.nn.LayerNorm): Layer normalization for the latent representation.
        linear_one (torch.nn.Linear): Linear layer to map latent representation to heatmap size.
        conv_list (torch.nn.ModuleList): List of convolutional layers for refining the heatmap.
        final_conv2d (torch.nn.Conv2d): Final convolutional layer to produce the output heatmap.
        context_encoder (LSTM_Encoder): LSTM-based encoder for context encoding.
    """
    
    def __init__(self, latent_dim, hm_size, encoder_arch, context_length):
        """
        Initializes the HeatmapDecoder.

        Args:
            latent_dim (int): Dimensionality of the latent representation.
            hm_size (int): Size of the heatmap (height and width).
            encoder_arch (dict): Architecture configuration for the LSTM encoder.
            context_length (int): Length of the context sequence.
        """
        super(HeatmapDecoder, self).__init__()
        self.hm_size = hm_size
        
        # Layernorm layer
        self.layernorm = torch.nn.LayerNorm(latent_dim)

        self.linear_one = torch.nn.Linear(latent_dim, hm_size * hm_size)
        self.conv_list = [torch.nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0)]
        self.conv_list.extend([torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0) for _ in range(5)])
        self.conv_list = torch.nn.ModuleList(self.conv_list)

        self.final_conv2d = torch.nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        X_arch = copy.deepcopy(encoder_arch)
        X_arch['pred_length'] = context_length
        X_arch['obs_length'] = context_length
            
        self.context_encoder = LSTM_Encoder(X_arch)

    def forward(self, x):
        """
        Forward pass of the HeatmapDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).

        Returns:
            torch.Tensor: Decoded heatmap of shape (batch_size, hm_size, hm_size).
        """
        # Encode the context using the LSTM encoder
        x = self.context_encoder(x[:, -3:])
        x = torch.nn.ELU()(x)      
        x = self.layernorm(x)   
        x = torch.nn.ELU()(self.linear_one(x))
        x = torch.nn.ELU()(x).reshape(-1, 1, self.hm_size, self.hm_size)

        # Pass through the convolutional layers
        for layer in self.conv_list:
            x = layer(x)
            x = torch.nn.ELU()(x)

        # Final convolution to produce the heatmap
        x = self.final_conv2d(x)
        
        return x.squeeze()