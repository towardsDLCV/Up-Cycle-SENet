import torch
import librosa

from data.dataset import display_spectrogram
from models.complex_nn import *
from data.conv_stft import *
from models.modules import ResidualBlock, ASPP


class Encoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super(Encoder, self).__init__()
        self.res = ResidualBlock(in_channels, out_channels)

        if downsample:
            self.conv = ComplexConv2d(out_channels, out_channels,
                                      kernel_size=(5, 7),
                                      stride=(2, 2),
                                      padding=(2, 3))
        else:
            self.conv = nn.Identity()

    def forward(self, inputs: torch.Tensor):
        outputs = self.res(inputs)
        outputs = self.conv(outputs)
        return outputs


class Decoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, upsample: bool, output_padding=(0, 0)):
        super(Decoder, self).__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels, out_channels)
        if upsample:
            self.conv = ComplexConvTranspose2d(out_channels, out_channels,
                                               kernel_size=(5, 7),
                                               stride=(2, 2),
                                               padding=(2, 3),
                                               output_padding=output_padding)
        else:
            self.conv = nn.Identity()

    def forward(self, inputs: torch.Tensor):
        outputs = self.res(inputs)
        outputs = self.conv(outputs)
        return outputs


############
# Generator
############
class Generator_C2N(nn.Module):

    def __init__(self, args, n_fft=512, hop_length=128):
        super(Generator_C2N, self).__init__()
        self.args = args
        self.n_fft = n_fft
        self.hop = hop_length

        self.stft = ConvSTFT(400, 100, 512, 'hanning', 'complex', fix=True)
        self.istft = ConviSTFT(400, 100, 512, 'hanning', 'complex', fix=True)

        # Mapping to Feature Space
        self.proj = ComplexConv2d(1, 16, kernel_size=(5, 7), padding=(2, 3))
        # Encoder
        self.d1 = Encoder(in_channels=16, out_channels=16, downsample=True)
        self.d2 = Encoder(in_channels=16, out_channels=16, downsample=True)
        self.d3 = Encoder(in_channels=16, out_channels=64, downsample=True)
        self.d4 = Encoder(in_channels=64, out_channels=256, downsample=False)
        # Bridge
        self.aspp1 = ASPP(in_channels=256, out_channels=256, atrous_rates=[3, 9])
        # Decoder
        self.u1 = Decoder(in_channels=256, out_channels=64, upsample=False)
        self.u2 = Decoder(in_channels=128, out_channels=16, upsample=True)
        self.u3 = Decoder(in_channels=32, out_channels=16, upsample=True, output_padding=(0, 1))
        self.u4 = Decoder(in_channels=32, out_channels=16, upsample=True)

        self.last = nn.Sequential(ASPP(in_channels=16, out_channels=16, atrous_rates=[3, 9]),
                                  ComplexConv2d(16, 16, kernel_size=(5, 7), padding=(2, 3)),
                                  ComplexBatchNorm2d(16),
                                  nn.LeakyReLU(inplace=True),
                                  ComplexConv2d(16, 1, kernel_size=(5, 7), padding=(2, 3)))

    def forward(self, x): # [B, T]
        x = x.unsqueeze(1)
        spec = self.stft(x)
        real = spec[:, :257]
        imag = spec[:, 257:]
        inputs = torch.stack([real, imag], dim=-1).unsqueeze(1)

        proj = self.proj(inputs)
        d1 = self.d1(proj)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        bridge = self.aspp1(d4)

        u1 = self.u1(bridge)
        c1 = torch.cat((u1, d3), dim=1)

        u2 = self.u2(c1)
        c2 = torch.cat((u2, d2), dim=1)

        u3 = self.u3(c2)
        c3 = torch.cat((u3, d1), dim=1)

        u4 = self.u4(c3)
        est = self.last(u4)

        real = est[..., 0]
        imag = est[..., 1]
        est = torch.cat([real, imag], dim=2).squeeze(1)
        output = self.istft(est).squeeze(1)
        output = torch.clamp_(output, -1, 1)

        return output
