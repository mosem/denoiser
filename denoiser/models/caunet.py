import math
import torch
import torch.nn as nn
from denoiser.models.modules import TorchSignalToFrames, TorchOLA, Dual_Transformer

from denoiser.resample import upsample2

from denoiser.utils import capture_init

import logging

logger = logging.getLogger(__name__)


class Dsconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=(1, 1)):
        super(Dsconv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    Dsconv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                             dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DecodeLayer(nn.Module):
    def __init__(self, dense_block, decode_block):
        super(DecodeLayer, self).__init__()
        self.dense_block = dense_block
        self.decode_block = decode_block

    def forward(self, x, skip):
        logger.info(f'decode layer input shape: {x.shape}')
        dense_out = self.dense_block(x)
        logger.info(f'dense_out shape: {x.shape}')
        cat_out = torch.cat([dense_out, skip], dim=1)
        logger.info(f'cat_out shape: {x.shape}')
        out = self.decode_block(cat_out)
        return out


class Caunet(nn.Module):

    @capture_init
    def __init__(self, frame_size=512, hidden=64, scale_factor=1, depth=4, dense_block_depth=3):
        super(Caunet, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = self.frame_size // 2
        self.depth= depth
        self.dense_block_depth = dense_block_depth
        # todo: remove redundant self parameters. or use them in code.
        self.N = 256
        self.B = 256
        self.H = 512
        self.P = 3
        # self.device = device
        self.in_channels = 1
        self.out_channels = 1
        self.kernel_size = (1, 3)
        self.stride = 2
        # self.elu = nn.SELU(inplace=True)
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.hidden = hidden
        self.scale_factor = scale_factor


        self.signalPreProcessor = TorchSignalToFrames(frame_size=self.frame_size,
                                                      frame_shift=self.frame_shift)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        tmp_frame_size = self.frame_size
        input_layer = [nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden, kernel_size=(1, 1)),
                       nn.LayerNorm(tmp_frame_size),
                       nn.PReLU(self.hidden)]
        self.encoder.append(nn.Sequential(*input_layer))
        for i in range(self.depth):
            encode_layer = [DenseBlock(tmp_frame_size, self.dense_block_depth, self.hidden),
                       self.pad1,
                       nn.Conv2d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=self.kernel_size,
                                 stride=(1,self.stride)),
                      nn.LayerNorm(math.ceil(tmp_frame_size / self.stride)),
                      nn.PReLU(self.hidden)]

            tmp_frame_size = math.ceil(tmp_frame_size / self.stride)

            dense_block = DenseBlock(tmp_frame_size, self.dense_block_depth, self.hidden)
            decode_block = nn.Sequential(SPConvTranspose2d(in_channels=self.hidden * 2, out_channels=self.hidden,
                                                           kernel_size=self.kernel_size,r=2),
                                         nn.LayerNorm(tmp_frame_size*self.stride),
                                         nn.PReLU(self.hidden))
            decode_layer = DecodeLayer(dense_block, decode_block)

            self.encoder.append(nn.Sequential(*encode_layer))
            self.decoder.insert(0, decode_layer)

            logger.info(f'{i}: {tmp_frame_size}')



        self.dual_transformer = Dual_Transformer(self.hidden, self.hidden, nhead=4,
                                                 num_layers=6)  # [batch, hidden, nframes, 8]

        self.dec_dense4 = DenseBlock(32, 3, self.hidden)
        self.dec_conv4 = SPConvTranspose2d(in_channels=self.hidden * 2, out_channels=self.hidden, kernel_size=self.kernel_size, r=2)
        self.dec_norm4 = nn.LayerNorm(64)
        self.dec_prelu4 = nn.PReLU(self.hidden)

        self.dec_dense3 = DenseBlock(64, 3, self.hidden)
        self.dec_conv3 = SPConvTranspose2d(in_channels=self.hidden * 2, out_channels=self.hidden, kernel_size=self.kernel_size, r=2)
        self.dec_norm3 = nn.LayerNorm(128)
        self.dec_prelu3 = nn.PReLU(self.hidden)

        self.dec_dense2 = DenseBlock(128, 3, self.hidden)
        self.dec_conv2 = SPConvTranspose2d(in_channels=self.hidden * 2, out_channels=self.hidden, kernel_size=self.kernel_size, r=2)
        self.dec_norm2 = nn.LayerNorm(256)
        self.dec_prelu2 = nn.PReLU(self.hidden)

        self.dec_dense1 = DenseBlock(256, 3, self.hidden)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.hidden * 2, out_channels=self.hidden, kernel_size=self.kernel_size, r=2)
        self.dec_norm1 = nn.LayerNorm(512)
        self.dec_prelu1 = nn.PReLU(self.hidden)

        self.out_conv = nn.Conv2d(in_channels=self.hidden, out_channels=self.out_channels, kernel_size=(1, 1))
        self.ola = TorchOLA(self.frame_shift)

    def estimate_valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.scale_factor)
        n_frames = math.ceil((length - self.frame_size) / self.frame_shift + 1)
        # todo: add pipeline of convolutions + dual transformer + transposed convolutions.
        length = (n_frames - 1) * self.frame_shift + self.frame_size
        return int(length)

    def forward(self, x):
        if self.scale_factor == 2:
            x = upsample2(x)
        elif self.scale_factor == 4:
            x = upsample2(x)
            x = upsample2(x)

        skips = []
        out = self.signalPreProcessor(x)

        for i, encode in enumerate(self.encoder):
            logger.info(f'encoder layer: {i}')
            out = encode(out)
            skips.append(out)

        # out = self.inp_prelu(self.inp_norm(self.inp_conv(out)))
        #
        # out = self.enc_dense1(out)
        # out = self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(out))))
        # skips.append(out)
        #
        # out = self.enc_dense2(out)
        # out = self.enc_prelu2(self.enc_norm2(self.enc_conv2(self.pad1(out))))
        # skips.append(out)
        #
        # out = self.enc_dense3(out)
        # out = self.enc_prelu3(self.enc_norm3(self.enc_conv3(self.pad1(out))))
        # skips.append(out)
        #
        # out = self.enc_dense4(out)
        # out = self.enc_prelu4(self.enc_norm4(self.enc_conv4(self.pad1(out))))
        # skips.append(out)

        out = self.dual_transformer(out)

        logger.info(f'dual transformer output shape: {out.shape}')

        # for i, decode in enumerate(self.decoder):
        #     logger.info(f'decoder layer: {i}')
        #     skip = skips.pop(-1)
        #     out = decode(out, skip)

        out = self.dec_dense4(out)
        out = torch.cat([out, skips[-1]], dim=1)
        out = self.dec_prelu4(self.dec_norm4(self.dec_conv4(self.pad1(out))))

        out = self.dec_dense3(out)
        out = torch.cat([out, skips[-2]], dim=1)
        out = self.dec_prelu3(self.dec_norm3(self.dec_conv3(self.pad1(out))))

        out = self.dec_dense2(out)
        out = torch.cat([out, skips[-3]], dim=1)
        out = self.dec_prelu2(self.dec_norm2(self.dec_conv2(self.pad1(out))))

        out = self.dec_dense1(out)
        out = torch.cat([out, skips[-4]], dim=1)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(out))))

        out = self.out_conv(out)
        out = self.ola(out)

        return out