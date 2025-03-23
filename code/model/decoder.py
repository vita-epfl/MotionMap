"""
This file is inspired from BeLFusion's codebase.
"""

import copy
import torch
from torch import nn
from base import BaseModel

# Model folder in main directory
from model.encoder import LSTM_Encoder
# Model folder in belfusion directory
from models.basics import rc, rc_recurrent


class ResidualRNNDecoder(nn.Module):
    """
    NOTE: This class is primarily unchanged. Or very small changes that I don't remember.
    A residual RNN decoder that supports LSTM and GRU cells. It optionally applies
    a non-linear input transformation (nin) and adds residual connections to the output.

    Args:
        n_in (int): Input feature size.
        n_out (int): Output feature size.
        n_hidden (int): Hidden state size of the RNN cell.
        rnn_type (str): Type of RNN cell to use ('lstm' or 'gru'). Default is 'lstm'.
        use_nin (bool): Whether to apply a non-linear input transformation. Default is False.
    """
    def __init__(
        self, n_in, n_out, n_hidden, rnn_type="lstm", use_nin=False
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.rnn_type = rnn_type
        self.use_nin = use_nin

        if self.rnn_type == "gru":
            self.rnn = nn.GRUCell(self.n_in, self.n_hidden)
            self.n_out = nn.Linear(self.n_hidden, self.n_out)
        else:
            self.rnn = nn.LSTMCell(self.n_in, self.n_hidden)
            self.n_out = nn.Linear(self.n_hidden, self.n_out)

        if self.use_nin:
            self.n_in = nn.Linear(self.n_in, self.n_in)


    def forward(self, x):
        """
        Forward pass of the decoder with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_in] or [batch_size, 1, n_in].

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor with residual connections applied.
                - torch.Tensor: Residual tensor (same as input).
        """
        if len(x.shape) == 3:
            x = x.squeeze(dim=1)
        elif len(x.shape) != 2:
            raise TypeError("invalid shape of tensor.")

        res = x
        if self.use_nin:
            x = self.n_in(x)

        if self.rnn_type == "lstm":
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden[0]
        else:
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden

        out = self.n_out(out_rnn)

        return out + res, res

    def forward_noresidual(self, x):
        """
        Forward pass of the decoder without residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_in].

        Returns:
            torch.Tensor: Output tensor without residual connections.
        """
        assert x.dim() == 2, "input must be 2D tensor"

        if self.rnn_type == "lstm":
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden[0]
        else:
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden

        return self.n_out(out_rnn)


class ResidualBehaviorNet(BaseModel):
    """
    A behavior forecasting model that uses residual connections, LSTM encoders and RNN-based decoders
    to predict future poses based on observed and contextual input sequences.

    Args:
        n_features (int): Number of features per landmark (e.g., 3 for x, y, z coordinates).
        n_landmarks (int): Number of keypoints in the pose.
        obs_length (int): Length of the observed sequence.
        pred_length (int): Length of the predicted sequence.
        context_length (int, optional): Length of the context sequence.
        hidden_dim (int, optional): Hidden dimension size for the encoder. Default is 16.
        dim_decoder_state (int, optional): Hidden state size for the decoder. Default is 128.
        decoder_arch (str, optional): Type of RNN cell to use in the decoder ('gru' or 'lstm'). Default is 'gru'.
        residual (bool, optional): Whether to use residual connections. Default is True.
        recurrent (bool, optional): Whether to use recurrent residual connections. Default is False.
        encoder_arch (dict, optional): Configuration for the encoder architecture. Default is None.
        projection (int, optional): Projection size for the decoder input.
    """
    
    def __init__(self, n_features, n_landmarks, obs_length, pred_length, context_length, projection,
        hidden_dim=16, dim_decoder_state=128, decoder_arch='gru', residual=True, recurrent=False,
        encoder_arch=None):
        
        super().__init__(n_features, n_landmarks, obs_length, pred_length)
        
        assert obs_length == pred_length, "obs_length != pred_length. Encoder can't be used for this case."

        # Decoder configuration
        self.n_kps = n_features * n_landmarks # 16 x 3 or 21 x 3
        self.residual = residual
        self.recurrent = recurrent

        self.dec_type = decoder_arch
        self.hidden_dim = hidden_dim
        self.projection = projection
        # Overwrite dim_decoder_state
        dim_decoder_state = projection * 2
        self.dim_decoder_state = projection * 2

        # Define the LSTM Encoder for Y
        self.encoder = LSTM_Encoder(encoder_arch)

        # Define the LSTM Encoder for X
        X_arch = copy.deepcopy(encoder_arch)
        X_arch['pred_length'] = context_length
        X_arch['obs_length'] = context_length
            
        self.context_encoder = LSTM_Encoder(X_arch)
        
        # Pose Forecaster
        print('Using ResidualRNNDecoder')
        self.decoder = ResidualRNNDecoder(
            n_in=dim_decoder_state,
            n_out= n_features * n_landmarks,
            n_hidden=dim_decoder_state,
            rnn_type=self.dec_type)
        
        # Define preprocessing on the concat of context_x and context_y
        self.encoder_preprocess = nn.Sequential(
            nn.Linear(self.dim_decoder_state, self.dim_decoder_state), nn.Tanh(),
            nn.Linear(self.dim_decoder_state, self.dim_decoder_state), nn.Tanh())


    def forward(self, x, y):
        """
        Forward pass to predict future poses based on observed and contextual input sequences.

        Args:
            x (torch.Tensor): Observed input tensor of shape [batch_size, obs_length, 1, n_landmarks, n_features].
            y (torch.Tensor): Ground truth tensor of shape [batch_size, pred_length, 1, n_landmarks, n_features].

        Returns:
            tuple: A tuple containing:
                - y_pred (torch.Tensor): Predicted poses of shape [batch_size, total_length, 1, n_landmarks, n_features].
                - hs (torch.Tensor): Hidden state tensor after preprocessing.
        """
        # x: [BS, 25 or 30, 1, 16 or 21, 3]
        # y: [BS, 100 or 120, 1, 16 or 21, 3]
        
        # [BS, length, participants, landmarks, features]
        batch_size, seq_len, n_agents, n_landmarks, n_features = x.shape
        
        # Context from Y
        context_y = self.encoder(y)

        # Context from X
        context_x = self.context_encoder(x)

        # Combined preprocessing for them
        hs = torch.cat([context_x, context_y], dim=1)
        hs = self.encoder_preprocess(hs)

        # y_pred placeholder: [BS, X+Y number of frames, 48 or 63]
        total_length = self.obs_length + x.shape[1]

        y_pred = torch.zeros([batch_size, total_length, self.n_kps], device=x.device)

        # Initialize hidden state of ResidualRNNDecoder to zero
        self.decoder.hidden = torch.zeros((x.shape[0], self.dim_decoder_state), device=x.device)

        for i in range(total_length):
            x_i = self.decoder.forward_noresidual(hs)
            y_pred[:, i] = x_i

        # What is the starting frame?
        last_obs = x[:, 0].view(batch_size, -1)
        
        if self.residual and self.recurrent:
            y_pred = rc_recurrent(last_obs, y_pred, batch_first=True) # prediction of offsets w.r.t. previous obs/prediction
        elif self.residual:
            y_pred = rc(last_obs, y_pred, batch_first=True) # prediction of offsets w.r.t. last obs (residual connection)
        
        y_pred = y_pred.reshape(
            (batch_size, total_length, 1, n_landmarks, n_features))
        
        return y_pred, hs.detach()


    def get_context_y(self, x=None, y=None):
        """
        Extracts the context from the ground truth sequence `y` and optionally predicts poses.

        Args:
            x (torch.Tensor, optional): Observed input tensor of shape [batch_size, obs_length, 1, n_landmarks, n_features].
            y (torch.Tensor): Ground truth tensor of shape [batch_size, pred_length, 1, n_landmarks, n_features].

        Returns:
            tuple: A tuple containing:
                - context_y (torch.Tensor): Encoded context from `y`.
                - y_pred (torch.Tensor or None): Predicted poses if `x` is provided, otherwise None.
        """
        context_y = self.encoder(y)
        y_pred = None
        
        if x is not None:
            y_pred = self.forward(x, y)

        return context_y, y_pred
    

    def get_context_x(self, x):
        """
        Extracts the context from the observed input sequence `x`.

        Args:
            x (torch.Tensor): Observed input tensor of shape [batch_size, obs_length, 1, n_landmarks, n_features].

        Returns:
            torch.Tensor: Encoded context from `x`.
        """
        context_x = self.context_encoder(x)

        return context_x
    

    def decode(self, x, projection):
        """
        Decodes the future poses based on the observed input sequence `x` and the provided projection.

        Args:
            x (torch.Tensor): Observed input tensor of shape [batch_size, obs_length, 1, n_landmarks, n_features].
            projection (torch.Tensor): Encoded projection tensor from the context of `y`.

        Returns:
            tuple: A tuple containing:
                - y_pred (torch.Tensor): Predicted poses of shape [batch_size, total_length, 1, n_landmarks, n_features].
                - hs (torch.Tensor): Hidden state tensor after preprocessing.
        """
        # x: [BS, 25 or 30, 1, 16 or 21, 3]
        # y: [BS, 100 or 120, 1, 16 or 21, 3]
        
        # [BS, length, participants, landmarks, features]
        batch_size, seq_len, n_agents, n_landmarks, n_features = x.shape
        
        # Context from Y's projection
        context_y = projection.detach()

        # Context from X
        context_x = self.context_encoder(x).detach()

        hs = torch.cat([context_x, context_y], dim=1)
        hs = self.encoder_preprocess(hs)
        
        # y_pred placeholder: [BS, 103 (or 123), 48 or 63]
        total_length = self.obs_length + x.shape[1]

        y_pred = torch.zeros([batch_size, total_length, self.n_kps], device=x.device)

        # Initialize hidden state of ResidualRNNDecoder to zero
        self.decoder.hidden = torch.zeros(
                (x.shape[0], self.dim_decoder_state), device=x.device)
        
        for i in range(total_length):
            x_i = self.decoder.forward_noresidual(hs)
            y_pred[:, i] = x_i

        # What is the starting frame?
        last_obs = x[:, 0].view(batch_size, -1)

        if self.residual and self.recurrent:
            y_pred = rc_recurrent(last_obs, y_pred, batch_first=True) # prediction of offsets w.r.t. previous obs/prediction
        elif self.residual:
            y_pred = rc(last_obs, y_pred, batch_first=True) # prediction of offsets w.r.t. last obs (residual connection)
        
        y_pred = y_pred.reshape(
            (batch_size, total_length, 1, n_landmarks, n_features))
        
        return y_pred, hs.detach()