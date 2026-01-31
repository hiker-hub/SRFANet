import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ConvNext.backbones.model_convnext import convnext_base
from .ConvNext.backbones.resnet import Resnet

from faf_module import FAFModule_Compatible
from rotated_conv import RotatedConvBlock


class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class MLP1D(nn.Module):
    """1D MLP for DSA module"""
    def __init__(self, in_channels, hid_channels, out_channels, norm_layer=None, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=False))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=False))
        self.proj = nn.Sequential(*mlps)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.proj(x)


class build_convnext(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False,
                 rotated_conv_kernel_size=3, conv_type='bottleneck'):
        super(build_convnext, self).__init__()
        
        self.return_f = return_f
        self.conv_type = conv_type

        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_base.fb_in22k_ft_in1k_384"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 1024
            self.convnext = convnext_base(pretrained=True, in_22k=True)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block

        conv_type_names = {
            'standard': 'Standard convolution',
            'bottleneck': 'Bottleneck convolution',
            'depthwise': 'Depthwise separable convolution',
            'multiscale': 'Multi-scale Inception convolution',
            'dilated': 'Dilated convolution'
        }
        
        print(f"\n{'=' * 70}")
        print(f"  improved convolution module")
        print(f"  Types: {conv_type_names.get(conv_type, conv_type)}")
        print(f"  Kernel size: {rotated_conv_kernel_size}x{rotated_conv_kernel_size}")
        print(f"  Number of blocks: {block}")
        print(f"{'=' * 70}\n")

        self.rot_conv_layer = RotatedConvBlock(
            in_channels=self.in_planes,
            kernel_size=rotated_conv_kernel_size,
            conv_type=conv_type
        )

        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

        in_channels = 1024
        hid_channels = 2048
        out_channels = 256
        num_layers = 2

        print("Using FAF module for domain alignment")
        self.FAF_module = FAFModule_Compatible(
            in_channels=in_channels,
            hid_channels=hid_channels,
            out_channels=out_channels,
            num_mlp=num_layers
        )

        self.scale = 1.
        self.l2_norm = True
        self.num_heads = 8

    def forward(self, x):
        gap_feature, part_features = self.convnext(x)
        if self.training:
            pfeat_align = self.FAF_module(part_features)
            enhanced_features = self.rot_conv_layer(part_features)
            tri_features = (enhanced_features, enhanced_features)
            convnext_feature = self.classifier1(gap_feature)
            tri_list = []
            for i in range(self.block):
                tri_list.append(tri_features[i % 2].mean([-2, -1]))
            triatten_features = torch.stack(tri_list, dim=2)
            if self.block == 0:
                y = []
            else:
                y = self.part_classifier(self.block, triatten_features, cls_name='classifier_mcb')
            y = y + [convnext_feature]
            classifier_feature_main = self.classifier1.add_block(gap_feature)
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return pfeat_align, cls, features, gap_feature, part_features, classifier_feature_main
            else:
                cls = [item for item in y]  
                features = None 
                return pfeat_align, cls, features, gap_feature, part_features, classifier_feature_main
        else:
            return gap_feature, part_features

    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y

def make_convnext_model(num_class, block=4, return_f=False, resnet=False,
                        rotated_conv_kernel_size=3, conv_type='bottleneck'):
    print('===========building convnext===========')
    
    model = build_convnext(
        num_class,
        block=block,
        return_f=return_f,
        resnet=resnet,
        rotated_conv_kernel_size=rotated_conv_kernel_size,
        conv_type=conv_type
    )
    return model

def make_model(config):
    conv_type = getattr(config, 'conv_type', 'bottleneck')
    
    print(f"\n[Model Configuration]")
    print(f"  Conv Type: {conv_type}")
    print(f"  Conv Kernel Size: {config.rotated_conv_kernel_size}")
    print(f"  Block Number: {config.block}")
    print(f"  Number of Classes: {config.nclasses}\n")
    
    model = make_convnext_model(
        num_class=config.nclasses,
        block=config.block,
        return_f=config.triplet_loss > 0,
        resnet=config.resnet,
        rotated_conv_kernel_size=config.rotated_conv_kernel_size,
        conv_type=conv_type
    )

    model.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    model.logit_scale_blocks = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config():
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }

    model.get_config = get_config
    return model
