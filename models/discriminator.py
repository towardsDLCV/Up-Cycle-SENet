import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from models.complex_nn import *
from data.conv_stft import *


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.convLayer_1 = nn.Sequential(
            nn.ReflectionPad1d(7),
            weight_norm(nn.Conv1d(1, 16, 15, stride=1)))

        self.convLayer_2 = nn.Sequential(
            weight_norm(nn.Conv1d(16, 64, 41, stride=4, padding=20, groups=4)))

        self.convLayer_3 = nn.Sequential(
            weight_norm(nn.Conv1d(64, 256, 41, stride=4, padding=20, groups=16)))

        self.convLayer_4 = nn.Sequential(
            weight_norm(nn.Conv1d(256, 1024, 41, stride=4, padding=20, groups=64)))

        self.convLayer_5 = nn.Sequential(
            weight_norm(nn.Conv1d(1024, 1024, 41, stride=4, padding=20, groups=256)))

        self.convLayer_6 = nn.Sequential(
            weight_norm(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)))

        self.convLayer_7 = weight_norm(nn.Conv1d(1024, 1, 3, stride=1, padding=1))

    def forward(self, inputs):
        x = inputs.unsqueeze(1)
        conv1 = self.convLayer_1(x)
        conv2 = self.convLayer_2(self.act(conv1))
        conv3 = self.convLayer_3(self.act(conv2))
        conv4 = self.convLayer_4(self.act(conv3))
        conv5 = self.convLayer_5(self.act(conv4))
        conv6 = self.convLayer_6(self.act(conv5))
        output = self.convLayer_7(self.act(conv6))
        return output

