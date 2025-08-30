import numpy as np
import math

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from einops import rearrange, repeat

from functools import partial
from fvcore.nn.flop_count import flop_count

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1, 125).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates,
                              inputs=interpolates,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def stft(x):
    return torch.stft(x, n_fft=512, hop_length=512 // 4, win_length=512,
                      center=True, normalized=False, onesided=True,
                      pad_mode='reflect',
                      window=torch.hann_window(512).cuda())


def ISTFT(x):
    return torch.istft(x, n_fft=512, hop_length=512//4, win_length=512,
                       center=True, normalized=False, onesided=True,
                       window=torch.hann_window(512).cuda(),
                       length=(x.size(2) - 1) * 512 // 4)


def split_last(x, shape):
    """split the last dimension to given shape"""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    """merge the last n_dims to a dimension"""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def gelu(x):
    """Implementation of the gelu activation function by Hugging Face"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# kernel functions

# transcribed from jax to pytorch from


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=False, eps=1e-4, device = None):
    #  Project_matrix: 정규기저벡터
    b, h, *_ = data.shape
    # data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)  # reshape
    projection = projection.type_as(data)

    #data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    data_dash = torch.einsum('...id,...jd->...ij',  data, projection)  # torch.matmul

    diag_data = data ** 2  # (2, 1, 1024, 64)
    diag_data = torch.sum(diag_data, dim=-1)  # (2, 1, 1024)
    diag_data = (diag_data / 2.0)
    diag_data = diag_data.unsqueeze(dim=-1)  # (2, 1, 1024, 1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data) + eps)
    """
    Paper:
    pi(U) = (1 / sqrt(m)) * exp(-||u||_2 / 2) * exp(Fu) 
    F : Gaussian Random Matrix
    U: Input featureMap
    """
    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)  # 정규분포 (dim_Head, dim_head=64)
    q, r = torch.qr(unstructured_block.cpu(), some=True)  # QR Decomposition
    q, r = map(lambda t: t.to(device), (q, r))  # Q(정규직교기저)
    return q.t()  # (64, 64)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)  # (nb_features, dim_heads)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix
# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    #  q, k ,v (batch, 1, 1024, 128)
    k_cumsum = k.sum(dim=-2)  # (2 , 1, 128)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # (2, 1, 1024)
    context = torch.einsum('...nd,...ne->...de', k, v)  # (2, 1, 128, 128)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)  # (2, 1, 1024, 128)
    return out


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


@torch.no_grad()
def moving_average_update(online, target, momentum):
    """
    Update target network parameters
    """
    for param_online, param_target in zip(online.parameters(), target.parameters()):
        param_target.data = param_target.data * momentum + param_online.data * (1. - momentum)


def calc_mean_std(feat, eps=1e-5):
    size = feat.data.size() # [B, N, H, W]
    assert (len(size) == 4)
    N, C = size[:2] # N: mini-batch / C: Channel
    feat_var = feat.view(N, C, -1).var(dim=2) + eps # [N, C, -1] -> [N, C, H*W]
    # torch.var(input, dim) dim: the dimension to reduce
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def get_flop_stats(model):
    """
    Compute the gflops for the current models given the config.
    Args:
        model (models): models to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given models.
    """
    inputs = (torch.randn(1, 32000).cuda(non_blocking=True))
    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

