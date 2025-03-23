import torch
import torch.nn as nn
from belfusion.utils.torch import *


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def reparametrize_logstd(mu, logstd):
    std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return eps.mul(std) + mu


def sample(mu):
    return torch.randn_like(mu)

def rc(x_start, pred, batch_first=True):
    # x_start -> [batch_size, ...]
    # pred -> [seq_length, batch_size, ...] | [batch_size, seq_length, ...]
    if batch_first:
        x_start = x_start.unsqueeze(1)
        shapes = [1 for s in x_start.shape]
        shapes[1] = pred.shape[1]
        x_start = x_start.repeat(shapes)
    else:
        x_start = x_start.unsqueeze(0)
        shapes = [1 for s in x_start.shape]
        shapes[0] = pred.shape[0]
        x_start = x_start.repeat(shapes)
    return x_start + pred

def rc_recurrent(x_start, pred, batch_first=True): # residual connection => offsets modeling
    # x_start -> [batch_size, ...]
    # pred -> [seq_length, batch_size, ...] | [batch_size, seq_length, ...]
    if batch_first:
        pred[:, 0] = x_start + pred[:, 0]
        for i in range(1, pred.shape[1]):
            pred[:, i] = pred[:, i-1] + pred[:, i]
    else: # seq length first
        pred[0] = x_start + pred[0]
        for i in range(1, pred.shape[0]):
            pred[i] = pred[i-1] + pred[i]
    return pred

class BasicMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], dropout=0.5, non_linearities='relu'):
        super(BasicMLP, self).__init__()
        self.non_linearities = non_linearities

        self.dropout = nn.Dropout(dropout)
        self.nl = nl[non_linearities]()

        self.denses = None
        
        # hidden dims
        hidden_dims = hidden_dims + [output_dim, ] # output dim is treated as the last hidden dim

        seqs = []
        for i in range(len(hidden_dims)):
            linear = nn.Linear(input_dim if i==0 else hidden_dims[i-1], hidden_dims[i])
            init_weights(linear)
            seqs.append(nn.Sequential(self.dropout, linear, self.nl))

        self.denses = nn.Sequential(*seqs)

    def forward(self, x):
        return self.denses(x) if self.denses is not None else x


class MLP(nn.Module):
    # https://github.com/Khrylx/DLow
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super(MLP, self).__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

class RNN(nn.Module):
    # https://github.com/Khrylx/DLow
    def __init__(self, input_dim, out_dim, cell_type='lstm', bi_dir=False):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = 'batch'
        rnn_cls = nn.LSTMCell if cell_type == 'lstm' else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == 'step':
            self.hx = zeros((batch_size, self.rnn_f.hidden_size)) if hx is None else hx
            if self.cell_type == 'lstm':
                self.cx = zeros((batch_size, self.rnn_f.hidden_size)) if cx is None else cx

    def forward(self, x):
        if self.mode == 'step':
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == 'lstm':
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == 'lstm':
            cx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == 'lstm':
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_out")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        # print("weights ", module)
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
    elif (
        isinstance(module, nn.LSTM)
        or isinstance(module, nn.RNN)
        or isinstance(module, nn.LSTMCell)
        or isinstance(module, nn.RNNCell)
        or isinstance(module, nn.GRU)
        or isinstance(module, nn.GRUCell)
    ):
        # https://www.cse.iitd.ac.in/~mausam/courses/col772/spring2018/lectures/12-tricks.pdf
        # • It can take a while for a RNN to learn to remember information
        # • Initialize biases for LSTM’s forget gate to 1 to remember more by default.
        # • Similarly, initialize biases for GRU’s reset gate to -1.
        DIV = 3 if isinstance(module, nn.GRU) or isinstance(module, nn.GRUCell) else 4
        for name, param in module.named_parameters():
            if "bias" in name:
                #print(name)
                nn.init.constant_(
                    param, 0.0
                )  
                if isinstance(module, nn.LSTMCell) \
                    or isinstance(module, nn.LSTM):
                    n = param.size(0)
                    # LSTM: (W_ii|W_if|W_ig|W_io), W_if (forget gate) => bias 1
                    start, end = n // DIV, n // 2
                    param.data[start:end].fill_(1.) # to remember more by default
                elif isinstance(module, nn.GRU) \
                    or isinstance(module, nn.GRUCell):
                    # GRU: (W_ir|W_iz|W_in), W_ir (reset gate) => bias -1
                    end = param.size(0) // DIV
                    param.data[:end].fill_(-1.) # to remember more by default
            elif "weight" in name:
                nn.init.xavier_normal_(param)
                if isinstance(module, nn.LSTMCell) \
                    or isinstance(module, nn.LSTM) \
                    or isinstance(module, nn.GRU) \
                    or isinstance(module, nn.GRUCell):
                    if 'weight_ih' in name: # input -> hidden weights
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                    elif 'weight_hh' in name: # hidden -> hidden weights (recurrent)
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul]) # orthogonal initialization https://arxiv.org/pdf/1702.00071.pdf
    else:
        print(f"[WARNING] Module not initialized: {module}")


nl = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "leaky_relu": nn.LeakyReLU,
    "none": lambda x: x,
}