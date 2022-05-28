import numpy as np
import functools
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.models import vgg16_bn
from torchvision.models.feature_extraction import create_feature_extractor


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class PixelShuffle_ICNR(nn.Sequential):
    def __init__(self, ni, nf, scale=2, blur=True, act_cls=nn.ReLU, spectral=False, norm_lyr=nn.BatchNorm2d):
        super().__init__()
        layers = [ConvNorm(ni, nf * (scale**2), ks=1, bn=False, act_cls=act_cls, spectral=spectral,
                           icnr=True, norm_lyr=norm_lyr),
                  nn.PixelShuffle(scale)]
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction, act_cls=nn.ReLU) -> None:
        super().__init__()
        nf = ch // reduction
        self.sq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNorm(ch, nf, ks=1, bn=False, act_cls=act_cls),
            ConvNorm(nf, ch, ks=1, bn=False, act_cls=nn.Sigmoid)
        )

    def forward(self, x):
        return x * self.sq(x)


class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.qkv_c = (n_channels // 8, n_channels // 8, n_channels)
        self.to_qkv = nn.Conv2d(n_channels, sum(self.qkv_c), kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):       # [B, C, H, W]
        size = x.size()
        qkv = self.to_qkv(x)
        q, k, v = qkv.flatten(2).split(self.qkv_c, dim=1)   # [B, (dq,dk,dv), H*W]
        attn = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)  # [B, lq, lk]
        o = torch.bmm(v, attn)
        o = o.view(*size)  # .contiguous()
        o = self.gamma * o + x
        return o


class ConvNorm(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, groups=1, bias=None, bn=True, bn_zero=False,
                 act_cls=nn.ReLU, norm_lyr=nn.BatchNorm2d, spectral=False, icnr=False):
        super().__init__()
        if padding is None:
            padding = 'same' if stride == 1 else int(np.ceil((ks - 1) / 2))
        if bias is None:
            bias = not bn
        while ni % groups:
            groups //= 2
        while nf % groups:
            groups //= 2
        self.conv = nn.Conv2d(ni, nf, ks, stride, padding, groups=groups, bias=bias)
        if icnr:
            self.conv.weight.data.copy_(icnr_init(self.conv.weight.data))
            self.conv.bias.data.zero_()
        if spectral:
            self.conv = spectral_norm(self.conv)
        if bn:
            self.bn = norm_lyr(nf)
            if bn_zero and norm_lyr is nn.BatchNorm2d:
                self.bn.weight.data.fill_(0.)
        else:
            self.bn = nn.Identity()
        if act_cls is None:
            self.act = nn.Identity()
        else:
            self.act = act_cls(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def model_sizes(encoder, size):
    with torch.no_grad():
        data = torch.randn(1, 3, *size, device=next(encoder.parameters()).device)
        out = encoder(data)
    return [i.shape for i in out]


class DynamicUnet(nn.Module):
    def __init__(self, encoder, n_inp, n_out, imsize=(256, 256), act_cls=nn.ReLU, norm_lyr=nn.InstanceNorm2d, spectral=True,
                 self_attn=False, blur=True):
        super().__init__()
        sizes = model_sizes(encoder, size=imsize)
        print(sizes)
        ni = sizes[-1][1]
        self.encoder = encoder
        middle_conv = nn.Sequential(ConvNorm(ni, ni * 2, act_cls=act_cls, norm_lyr=norm_lyr),
                                    ConvNorm(ni * 2, ni, act_cls=act_cls, norm_lyr=norm_lyr))
        self.mid = nn.Sequential(norm_lyr(ni), act_cls(), middle_conv)
        self.unet_blocks = nn.ModuleList()
        sizes = sizes[:-1]
        for i, sz in enumerate(sizes[::-1]):
            not_final = i != len(sizes) - 1
            sa = self_attn and (i == len(sizes) - 3)
            nim = ni // 2 + sz[1]
            nf = nim if not_final else nim // 2
            unet_block = UnetBlock(ni, nf, sz[1], blur=blur, self_attn=sa,
                                   act_cls=act_cls, spectral=spectral, norm_lyr=norm_lyr)
            self.unet_blocks.append(unet_block)
            ni = nf
        self.pix_up = PixelShuffle_ICNR(ni, ni, act_cls=act_cls, norm_lyr=norm_lyr)
        ni += n_inp
        self.last_cross = ResBlock(ni, ni, act_cls=act_cls, norm_lyr=norm_lyr)
        self.conv_out = ConvNorm(ni, n_out, ks=1, act_cls=None, bn=False)

    def forward(self, inp, get_feat=False, encode_only=False):
        feats = self.encoder(inp)
        x = self.mid(feats[-1])
        for ft, lyr in zip(feats[:-1][::-1], self.unet_blocks):
            x = lyr(x, ft)
        x = self.pix_up(x)
        x = torch.cat([x, inp], dim=1)
        x = self.last_cross(x)
        x = self.conv_out(x)
        if get_feat:
            feats = [inp] + feats
            if encode_only:
                return feats
            else:
                return torch.tanh(x), feats
        return torch.tanh(x)


class UnetBlock(nn.Module):
    def __init__(self, ni, nf, skip_in, blur=True, act_cls=nn.ReLU,
                 self_attn=False, spectral=True, norm_lyr=nn.BatchNorm2d):
        super().__init__()
        self.pix_shuf = PixelShuffle_ICNR(ni, ni // 2, blur=blur, act_cls=act_cls,
                                          spectral=spectral, norm_lyr=norm_lyr)
        rin = ni // 2 + skip_in
        self.resb = ResBlock(rin, nf, spectral=spectral, act_cls=act_cls, self_attn=self_attn, norm_lyr=norm_lyr)
        self.act = act_cls()
        self.bn = norm_lyr(skip_in)

    def forward(self, x, skip=None):
        x = self.pix_shuf(x)
        # x = F.interpolate(x, skip.shape[-2:], mode='nearest')
        if skip is not None:
            x = self.act(torch.cat([x, self.bn(skip)], dim=1))
        return self.resb(x)


class ResBlock(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=1, groups=1, reduction=0, spectral=False,
                 act_cls=nn.ReLU, self_attn=False, norm_lyr=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = ConvNorm(ni, nf, ks, stride, groups=groups, act_cls=act_cls, spectral=spectral,
                              norm_lyr=norm_lyr)
        self.conv2 = ConvNorm(nf, nf, ks, groups=1, act_cls=None, spectral=spectral, bn_zero=True,
                              norm_lyr=norm_lyr)
        self.act = act_cls(inplace=True) if act_cls is nn.ReLU else act_cls()

        shortcut = []
        if ni != nf:
            shortcut.append(ConvNorm(ni, nf, 1, act_cls=nn.Identity, norm_lyr=norm_lyr))
        if stride > 1:
            shortcut.append(nn.MaxPool2d(stride))
        self.shortcut = nn.Sequential(*shortcut)

        if self_attn:
            self.atn = SelfAttention(nf)
        elif reduction:
            self.atn = SqueezeExcite(nf, reduction)
        else:
            self.atn = nn.Identity()

    def forward(self, x):
        inp = x
        x = self.conv2(self.conv1(x))
        x = self.atn(x)
        return self.act(x + self.shortcut(inp))


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.use_mlp = use_mlp

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches=256, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    patch_id = torch.randperm(feat_reshape.shape[1], dtype=torch.long, device=feat.device)
                    # patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                # patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            if isinstance(patch_id, np.ndarray):
                patch_id = torch.tensor(patch_id, device=feats[0].device)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False, gauss_std=0.1):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [GaussianNoise(gauss_std), nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                                            stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [GaussianNoise(gauss_std),
                        nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                        nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    GaussianNoise(gauss_std),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    GaussianNoise(gauss_std),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            GaussianNoise(gauss_std),
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            GaussianNoise(gauss_std),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training and self.std != 0.:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super().__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, inp, target_is_real):
        target_tensor = torch.empty_like(inp).fill_(target_is_real)
        return self.loss(inp, target_tensor)


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


class VGGLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer_ids = [22, 32, 42]
        self.weights = [5, 15, 4]
        m = vgg16_bn(pretrained=True).features.eval()
        return_nodes = {f'{x}': f'feat{i}' for i, x in enumerate(layer_ids)}
        self.vgg_fx = create_feature_extractor(m, return_nodes=return_nodes)
        self.vgg_fx.requires_grad_(False)
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg_fx(x)
        with torch.inference_mode():
            y_vgg = self.vgg_fx(y)
        loss = self.l1_loss(x, y)
        for i, k in enumerate(x_vgg.keys()):
            loss += self.weights[i] * self.l1_loss(x_vgg[k], y_vgg[k].detach_())       # feature loss
            loss += self.weights[i]**2 * 5e3 * self.l1_loss(gram_matrix(x_vgg[k]), gram_matrix(y_vgg[k]))  # style loss
        return loss


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
