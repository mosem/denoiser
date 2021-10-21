import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import GRU
from torch.nn.modules.normalization import LayerNorm

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


class TorchSignalToFrames(object):
    r"""Chunks a tensor into frames
         required input shape is [1, 1, -1]
         input params:    (frame_size: window_size,  frame_shift: overlap(samples))
         output:   [1, 1, num_frames, frame_size]
    """

    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a



class TorchOLA(nn.Module):
    r"""Performs overlap-and-add on gpu using torch tensor
        required input is tensor
        perform frames into signal
        used in the output of network
    """

    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device,
                          requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = GRU(d_model, d_model, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        if bidirectional:
            # self.linear2 = Linear(d_model*2*2, d_model)
            self.linear2 = GRU(d_model * 2, d_model, bidirectional=False)
        else:
            self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        self.linear2.flatten_parameters()
        src2, h_n1 = self.linear2(self.dropout(self.activation(out)))
        del h_n1
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Dual_Transformer(nn.Module):
    """
    Deep duaL-path RNN.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, input_size, output_size, nhead=4, dropout=0, num_layers=1):
        super(Dual_Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=nhead, dropout=dropout, bidirectional=True))
            self.col_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=nhead, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size // 2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        # output = input
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)  # [dim1, b*dim2, c]
            row_output = self.row_trans[i](row_input)  # [dim1, b*dim2, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [b, c, dim2, dim1]
            row_output = self.row_norm[i](row_output)  # [b, c, dim2, dim1]
            output = output + row_output  # [b, c, dim2, dim1]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)  # [dim2, b*dim1, c]
            col_output = self.col_trans[i](col_input)  # [dim2, b*dim1, c]
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [b, c, dim2, dim1]
            col_output = self.col_norm[i](col_output)  # [b, c, dim2, dim1]
            output = output + col_output  # [b, c, dim2, dim1]

        del row_input, row_output, col_input, col_output
        output = self.output(output)  # [b, c, dim2, dim1]

        return output