import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=groups, bias=bias, **kwargs)
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.imag_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        complex_real = self.real_conv(x_real) - self.imag_conv(x_imag)
        complex_imag = self.imag_conv(x_real) + self.real_conv(x_imag)

        output = torch.stack([complex_real, complex_imag], dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        self.real_Transconv = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 output_padding=output_padding,
                                                 groups=groups,
                                                 bias=bias,
                                                 dilation=dilation,
                                                 **kwargs)
        self.imag_Transconv = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 output_padding=output_padding,
                                                 groups=groups,
                                                 bias=bias,
                                                 dilation=dilation,
                                                 **kwargs)
        nn.init.xavier_uniform_(self.real_Transconv.weight)
        nn.init.xavier_uniform_(self.imag_Transconv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        complex_real = self.real_Transconv(x_real) - self.imag_Transconv(x_imag)
        complex_imag = self.imag_Transconv(x_real) + self.real_Transconv(x_imag)

        output = torch.stack([complex_real, complex_imag], dim=-1)

        return output


class ComplexDepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.bn = ComplexBatchNorm2d(in_channels)
        self.conv_3x3 = ComplexConv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, groups=in_channels, bias=bias, **kwargs)
        self.conv_1x1 = ComplexConv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        output = self.conv_3x3(x)
        output = self.bn(output)
        output = self.conv_1x1(output)
        return output


def ComplexInterpolate(x, size):
    x_real = x[..., 0]
    x_imag = x[..., 1]

    complex_real = F.interpolate(x_real, size=size, mode="bilinear", align_corners=False)
    complex_imag = F.interpolate(x_imag, size=size, mode="bilinear", align_corners=False)

    output = torch.stack([complex_real, complex_imag], dim=-1)
    return output


class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.real = nn.AdaptiveAvgPool2d(size)
        self.imag = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        complex_real = self.real(x_real)
        complex_imag = self.imag(x_imag)

        output = torch.stack([complex_real, complex_imag], dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class ComplexInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super(ComplexInstanceNorm2d, self).__init__()
        self.bn_re = nn.InstanceNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.InstanceNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super(ComplexLayerNorm, self).__init__()
        self.real = LayerNorm(dim)
        self.imag = LayerNorm(dim)

    def forward(self, x):
        real = self.real(x[..., 0])
        imag = self.imag(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class ComplexReflectionPad2d(nn.Module):

    def __init__(self, size):
        super(ComplexReflectionPad2d, self).__init__()
        self.real = nn.ReflectionPad2d(size)
        self.imag = nn.ReflectionPad2d(size)

    def forward(self, inputs):
        real = inputs[..., 0]
        imag = inputs[..., 1]

        real_pad = self.real(real)
        imag_pad = self.imag(imag)

        outputs = torch.stack([real_pad, imag_pad], dim=-1)
        return outputs
