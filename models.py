import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

import utils


class FCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.last = nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.last(x)
        return x


class Wakey_FCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(Wakey_FCN, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.last = nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x


class Single_FCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(Single_FCN, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.last = nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x


def get_unet():
    return smp.Unet(
        encoder_name='resnet18', encoder_depth=3, encoder_weights=None,
        decoder_channels=(128, 64, 64), in_channels=4, classes=2)


def get_unet_plus_plus():
    return smp.UnetPlusPlus(
        encoder_name='resnet18', encoder_depth=3, encoder_weights=None,
        decoder_channels=(128, 64, 64), in_channels=4, classes=2
    )


def get_manet():
    return smp.MAnet(
        encoder_name='resnet18', encoder_depth=3, encoder_weights=None,
        decoder_channels=(128, 64, 64), in_channels=4, classes=2
    )


def get_deeplabv3():
    return smp.DeepLabV3(
        encoder_name='resnet18', encoder_depth=3, encoder_weights=None,
        decoder_channels=128, in_channels=4, classes=2
    )


def get_deeplabv3_plus():
    return smp.DeepLabV3Plus(encoder_name='resnet34', encoder_depth=5, encoder_weights=None,
                             encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
                             in_channels=4, classes=2)


def get_fpn():
    return smp.FPN(encoder_name='resnet34', encoder_depth=5, encoder_weights=None, decoder_pyramid_channels=256,
                   decoder_segmentation_channels=128, decoder_merge_policy='add', in_channels=4,
                   classes=2)


def get_linknet():
    return smp.Linknet(encoder_name='resnet18', encoder_depth=3, encoder_weights=None, in_channels=4,
                       classes=2)


def get_pspnet():
    return smp.PSPNet(encoder_name='resnet34', encoder_weights=None, encoder_depth=3, psp_out_channels=512,
                      psp_use_batchnorm=True, psp_dropout=0.2, in_channels=4, classes=2)


def get_pan():
    return smp.PAN(encoder_name='resnet34', encoder_weights=None, encoder_dilation=True, decoder_channels=32,
                   in_channels=4, classes=2)


def get_fcn():
    return FCN(num_input_channels=4, num_output_classes=len(utils.NLCD_CLASSES), num_filters=64)


def get_water_fcn():
    return Single_FCN(num_input_channels=1, num_output_classes=2, num_filters=64)


def get_imprev_fcn():
    return Single_FCN(num_input_channels=4, num_output_classes=2, num_filters=64)


def get_wakey_fcn():
    return Wakey_FCN(num_input_channels=1, num_output_classes=2, num_filters=64)
