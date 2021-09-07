# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/dcgan.py


# from utils.model_ops import *
from utils.model_ops import sndeconv2d, deconv2d, ConditionalBatchNorm2d, batchnorm_2d, snlinear, linear, Self_Attn, snconv2d, conv2d, sn_embedding, embedding, init_weights
# from utils.misc import *
from utils.misc import dummy_context_mgr

# Quaternion inputs
from utils.model_qops import qsndeconv2d, midqsndeconv2d, qdeconv2d, qbatchnorm_2d, qsnlinear, midqsnlinear, qlinear, QSelf_Attn, qsnconv2d, midqsnconv2d, qconv2d, MidQSelf_Attn

import torch
import torch.nn as nn
import torch.nn.functional as F



class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, num_classes):
        super(GenBlock, self).__init__()
        self.conditional_bn = conditional_bn

        if g_spectral_norm:
            self.deconv0 = sndeconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)
        else:
            self.deconv0 = deconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)

        if self.conditional_bn:
            self.bn0 = ConditionalBatchNorm2d(num_features=out_channels, num_classes=num_classes,
                                              spectral_norm=g_spectral_norm)
        else:
            self.bn0 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x, label):
        x = self.deconv0(x)
        if self.conditional_bn:
            x = self.bn0(x, label)
        else:
            x = self.bn0(x)
        out = self.activation(x)
        return out


class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, img_size, channel, g_conv_dim, g_spectral_norm, attention, attention_after_nth_gen_block,
                 conditional_strategy, num_classes, needs_init, mixed_precision):
        super(Generator, self).__init__()
        self.in_dims =  [512, 256, 128]
        self.out_dims = [256, 128, 64]

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        conditional_bn = True if conditional_strategy in ["ACGAN", "ProjGAN", "ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"] else False

        if g_spectral_norm:
            self.linear0 = snlinear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)
        else:
            self.linear0 = linear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                          out_channels=self.out_dims[index],
                                          g_spectral_norm=g_spectral_norm,
                                          activation_fn="ReLU",
                                          conditional_bn=conditional_bn,
                                          num_classes=self.num_classes)]]

            if index+1 == attention_after_nth_gen_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if g_spectral_norm:
            self.conv4 = snconv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv4 = conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if needs_init is not False:
            init_weights(self.modules)

    def forward(self, z, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], 4, 4)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, Self_Attn):
                        act = block(act)
                    else:
                        act = block(act, label)
            act = self.conv4(act)
            out = self.tanh(act)
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn="ReLU"):
        super(DiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm

        if d_spectral_norm:
            self.conv0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

            self.bn0 = batchnorm_2d(in_features=out_channels)
            self.bn1 = batchnorm_2d(in_features=out_channels)

        activation_fn = "ReLU"

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv0(x)
        if self.d_spectral_norm is False:
            x = self.bn0(x)
        x = self.activation(x)
        x = self.conv1(x)
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        out = self.activation(x)
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_size, d_conv_dim, d_spectral_norm, attention, attention_after_nth_dis_block, conditional_strategy,
                 hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, needs_init, channel, mixed_precision):
        super(Discriminator, self).__init__()
        self.in_dims  = [3] + [64, 128]
        self.out_dims = [64, 128, 256]

        self.d_spectral_norm = d_spectral_norm
        self.conditional_strategy = conditional_strategy
        self.num_classes = num_classes
        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.mixed_precision = mixed_precision

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
                                       out_channels=self.out_dims[index],
                                       d_spectral_norm=d_spectral_norm)]]

            if index+1 == attention_after_nth_dis_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], d_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if self.d_spectral_norm:
            self.conv = snconv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        else:
            self.conv = conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.bn = batchnorm_2d(in_features=512)


        activation_fn = "ReLU"


        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError


        if d_spectral_norm:
            self.linear1 = snlinear(in_features=512, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = snlinear(in_features=512, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = snlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = sn_embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = sn_embedding(num_classes, 512)
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = snlinear(in_features=512, out_features=num_classes)
            else:
                pass
        else:
            self.linear1 = linear(in_features=512, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = linear(in_features=512, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = linear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = embedding(num_classes, 512)
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = linear(in_features=512, out_features=num_classes)
            else:
                pass

        # Weight init
        if needs_init is not False:
            init_weights(self.modules)


    def forward(self, x, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.conv(h)
            if self.d_spectral_norm is False:
                h = self.bn(h)
            h = self.activation(h)
            h = torch.sum(h, dim=[2, 3])

            if self.conditional_strategy == 'no':
                authen_output = torch.squeeze(self.linear1(h))
                return authen_output

            elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                authen_output = torch.squeeze(self.linear1(h))
                cls_proxy = self.embedding(label)
                cls_embed = self.linear2(h)
                if self.nonlinear_embed:
                    cls_embed = self.linear3(self.activation(cls_embed))
                if self.normalize_embed:
                    cls_proxy = F.normalize(cls_proxy, dim=1)
                    cls_embed = F.normalize(cls_embed, dim=1)
                return cls_proxy, cls_embed, authen_output

            elif self.conditional_strategy == 'ProjGAN':
                authen_output = torch.squeeze(self.linear1(h))
                proj = torch.sum(torch.mul(self.embedding(label), h), 1)
                return authen_output + proj

            elif self.conditional_strategy == 'ACGAN':
                authen_output = torch.squeeze(self.linear1(h))
                cls_output = self.linear4(h)
                return cls_output, authen_output

            else:
                raise NotImplementedError


####################
# Quaternion DCGAN #
####################


class QuaternionGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, num_classes):
        super(QuaternionGenBlock, self).__init__()
        self.conditional_bn = conditional_bn

        if g_spectral_norm:
            self.deconv0 = qsndeconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)
        else:
            self.deconv0 = qdeconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)

        if self.conditional_bn:
            self.bn0 = ConditionalBatchNorm2d(num_features=out_channels, num_classes=num_classes,
                                              spectral_norm=g_spectral_norm)
        else:
            self.bn0 = qbatchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x, label):
        x = self.deconv0(x)
        if self.conditional_bn:
            x = self.bn0(x, label)
        else:
            x = self.bn0(x)
        out = self.activation(x)
        return out


class QuaternionGenerator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, img_size, channel, g_conv_dim, g_spectral_norm, attention, attention_after_nth_gen_block,
                 conditional_strategy, num_classes, needs_init, mixed_precision):
        super(QuaternionGenerator, self).__init__()
        self.in_dims =  [512, 256, 128]
        self.out_dims = [256, 128, 64]

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        conditional_bn = True if conditional_strategy in ["ACGAN", "ProjGAN", "ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"] else False

        if g_spectral_norm:
            self.linear0 = qsnlinear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)
        else:
            self.linear0 = qlinear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[QuaternionGenBlock(in_channels=self.in_dims[index],
                                          out_channels=self.out_dims[index],
                                          g_spectral_norm=g_spectral_norm,
                                          activation_fn="ReLU",
                                          conditional_bn=conditional_bn,
                                          num_classes=self.num_classes)]]

            if index+1 == attention_after_nth_gen_block and attention is True:
                self.blocks += [[QSelf_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if g_spectral_norm:
            self.conv4 = qsnconv2d(in_channels=self.out_dims[-1], out_channels=4, kernel_size=3, stride=1, padding=1)
        else:
            self.conv4 = qconv2d(in_channels=self.out_dims[-1], out_channels=4, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if needs_init is not False:
            init_weights(self.modules)

    def forward(self, z, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], 4, 4)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, QSelf_Attn):
                        act = block(act)
                    else:
                        act = block(act, label)
            act = self.conv4(act)
            out = self.tanh(act)
        return out


class QuaternionDiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn="ReLU"):
        super(QuaternionDiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm

        if d_spectral_norm:
            self.conv0 = qsnconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = qsnconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv0 = qconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = qconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

            self.bn0 = qbatchnorm_2d(in_features=out_channels)
            self.bn1 = qbatchnorm_2d(in_features=out_channels)

        activation_fn = "ReLU"

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv0(x)
        if self.d_spectral_norm is False:
            x = self.bn0(x)
        x = self.activation(x)
        x = self.conv1(x)
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        out = self.activation(x)
        return out


class QuaternionDiscriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_size, d_conv_dim, d_spectral_norm, attention, attention_after_nth_dis_block, conditional_strategy,
                 hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, needs_init, channel, mixed_precision):
        super(QuaternionDiscriminator, self).__init__()
        self.in_dims  = [channel] + [64, 128]
        self.out_dims = [64, 128, 256]

        self.d_spectral_norm = d_spectral_norm
        self.conditional_strategy = conditional_strategy
        self.num_classes = num_classes
        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.mixed_precision = mixed_precision

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[QuaternionDiscBlock(in_channels=self.in_dims[index],
                                       out_channels=self.out_dims[index],
                                       d_spectral_norm=d_spectral_norm)]]

            if index+1 == attention_after_nth_dis_block and attention is True:
                self.blocks += [[QSelf_Attn(self.out_dims[index], d_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if self.d_spectral_norm:
            self.conv = qsnconv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        else:
            self.conv = qconv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.bn = qbatchnorm_2d(in_features=512)


        activation_fn = "ReLU"


        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError


        if d_spectral_norm:
            self.linear1 = linear(in_features=512, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = qsnlinear(in_features=512, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = qsnlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = sn_embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = sn_embedding(num_classes, 512)
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = qsnlinear(in_features=512, out_features=num_classes)
            else:
                pass
        else:
            self.linear1 = linear(in_features=512, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = qlinear(in_features=512, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = qlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = embedding(num_classes, 512)
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = qlinear(in_features=512, out_features=num_classes)
            else:
                pass

        # Weight init
        if needs_init is not False:
            init_weights(self.modules)


    def forward(self, x, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.conv(h)
            if self.d_spectral_norm is False:
                h = self.bn(h)
            h = self.activation(h)
            h = torch.sum(h, dim=[2, 3])

            if self.conditional_strategy == 'no':
                authen_output = torch.squeeze(self.linear1(h))
                return authen_output

            elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                authen_output = torch.squeeze(self.linear1(h))
                cls_proxy = self.embedding(label)
                cls_embed = self.linear2(h)
                if self.nonlinear_embed:
                    cls_embed = self.linear3(self.activation(cls_embed))
                if self.normalize_embed:
                    cls_proxy = F.normalize(cls_proxy, dim=1)
                    cls_embed = F.normalize(cls_embed, dim=1)
                return cls_proxy, cls_embed, authen_output

            elif self.conditional_strategy == 'ProjGAN':
                authen_output = torch.squeeze(self.linear1(h))
                proj = torch.sum(torch.mul(self.embedding(label), h), 1)
                return authen_output + proj

            elif self.conditional_strategy == 'ACGAN':
                authen_output = torch.squeeze(self.linear1(h))
                cls_output = self.linear4(h)
                return cls_output, authen_output

            else:
                raise NotImplementedError


########################
# Mid Quaternion DCGAN #
########################


class MidQuaternionGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, num_classes):
        super(MidQuaternionGenBlock, self).__init__()
        self.conditional_bn = conditional_bn

        if g_spectral_norm:
            self.deconv0 = midqsndeconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)
        else:
            self.deconv0 = qdeconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)

        if self.conditional_bn:
            self.bn0 = ConditionalBatchNorm2d(num_features=out_channels, num_classes=num_classes,
                                              spectral_norm=g_spectral_norm)
        else:
            self.bn0 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x, label):
        x = self.deconv0(x)
        if self.conditional_bn:
            x = self.bn0(x, label)
        else:
            x = self.bn0(x)
        out = self.activation(x)
        return out


class MidQuaternionGenerator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, img_size, channel, g_conv_dim, g_spectral_norm, attention, attention_after_nth_gen_block,
                 conditional_strategy, num_classes, needs_init, mixed_precision):
        super(MidQuaternionGenerator, self).__init__()
        self.in_dims =  [512, 256, 128]
        self.out_dims = [256, 128, 64]

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        conditional_bn = True if conditional_strategy in ["ACGAN", "ProjGAN", "ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"] else False

        if g_spectral_norm:
            self.linear0 = snlinear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)
        else:
            self.linear0 = linear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[MidQuaternionGenBlock(in_channels=self.in_dims[index],
                                          out_channels=self.out_dims[index],
                                          g_spectral_norm=g_spectral_norm,
                                          activation_fn="ReLU",
                                          conditional_bn=conditional_bn,
                                          num_classes=self.num_classes)]]

            if index+1 == attention_after_nth_gen_block and attention is True:
                self.blocks += [[MidQSelf_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if g_spectral_norm:
            self.conv4 = midqsnconv2d(in_channels=self.out_dims[-1], out_channels=4, kernel_size=3, stride=1, padding=1)
        else:
            self.conv4 = qconv2d(in_channels=self.out_dims[-1], out_channels=4, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if needs_init is not False:
            init_weights(self.modules)

    def forward(self, z, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], 4, 4)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, MidQSelf_Attn):
                        act = block(act)
                    else:
                        act = block(act, label)
            act = self.conv4(act)
            out = self.tanh(act)
        return out


class MidQuaternionDiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn="ReLU"):
        super(MidQuaternionDiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm

        if d_spectral_norm:
            self.conv0 = midqsnconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = midqsnconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv0 = qconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = qconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

            self.bn0 = batchnorm_2d(in_features=out_channels)
            self.bn1 = batchnorm_2d(in_features=out_channels)

        activation_fn = "ReLU"

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv0(x)
        if self.d_spectral_norm is False:
            x = self.bn0(x)
        x = self.activation(x)
        x = self.conv1(x)
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        out = self.activation(x)
        return out


class MidQuaternionDiscriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_size, d_conv_dim, d_spectral_norm, attention, attention_after_nth_dis_block, conditional_strategy,
                 hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, needs_init, channel, mixed_precision):
        super(MidQuaternionDiscriminator, self).__init__()
        self.in_dims  = [channel] + [64, 128]
        self.out_dims = [64, 128, 256]

        self.d_spectral_norm = d_spectral_norm
        self.conditional_strategy = conditional_strategy
        self.num_classes = num_classes
        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.mixed_precision = mixed_precision

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[MidQuaternionDiscBlock(in_channels=self.in_dims[index],
                                       out_channels=self.out_dims[index],
                                       d_spectral_norm=d_spectral_norm)]]

            if index+1 == attention_after_nth_dis_block and attention is True:
                self.blocks += [[MidQSelf_Attn(self.out_dims[index], d_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if self.d_spectral_norm:
            self.conv = midqsnconv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        else:
            self.conv = qconv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.bn = batchnorm_2d(in_features=512)


        activation_fn = "ReLU"


        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError


        if d_spectral_norm:
            self.linear1 = linear(in_features=512, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = midqsnlinear(in_features=512, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = midqsnlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = sn_embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = sn_embedding(num_classes, 512)
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = midqsnlinear(in_features=512, out_features=num_classes)
            else:
                pass
        else:
            self.linear1 = linear(in_features=512, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = qlinear(in_features=512, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = qlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = embedding(num_classes, 512)
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = qlinear(in_features=512, out_features=num_classes)
            else:
                pass

        # Weight init
        if needs_init is not False:
            init_weights(self.modules)


    def forward(self, x, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.conv(h)
            if self.d_spectral_norm is False:
                h = self.bn(h)
            h = self.activation(h)
            h = torch.sum(h, dim=[2, 3])

            if self.conditional_strategy == 'no':
                authen_output = torch.squeeze(self.linear1(h))
                return authen_output

            elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                authen_output = torch.squeeze(self.linear1(h))
                cls_proxy = self.embedding(label)
                cls_embed = self.linear2(h)
                if self.nonlinear_embed:
                    cls_embed = self.linear3(self.activation(cls_embed))
                if self.normalize_embed:
                    cls_proxy = F.normalize(cls_proxy, dim=1)
                    cls_embed = F.normalize(cls_embed, dim=1)
                return cls_proxy, cls_embed, authen_output

            elif self.conditional_strategy == 'ProjGAN':
                authen_output = torch.squeeze(self.linear1(h))
                proj = torch.sum(torch.mul(self.embedding(label), h), 1)
                return authen_output + proj

            elif self.conditional_strategy == 'ACGAN':
                authen_output = torch.squeeze(self.linear1(h))
                cls_output = self.linear4(h)
                return cls_output, authen_output

            else:
                raise NotImplementedError
