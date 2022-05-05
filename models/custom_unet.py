import numpy as np

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
        return self.act(x.add_(self.shortcut(inp)))


class UnetBlock(nn.Module):
    def __init__(self, ni, nf, skip_in, blur=True, act_cls=nn.ReLU, groups=1,
                 self_attn=False, reduction=0, spectral=False, norm_lyr=nn.BatchNorm2d):
        super().__init__()
        self.pix_shuf = PixelShuffle_ICNR(ni, ni // 2, blur=blur, act_cls=act_cls,
                                          spectral=spectral, norm_lyr=norm_lyr)
        rin = ni // 2 + skip_in
        self.resb = ResBlock(rin, nf, groups=groups, reduction=reduction, spectral=spectral,
                             act_cls=act_cls, self_attn=self_attn, norm_lyr=norm_lyr)

    def forward(self, x, skip=None):
        x = self.pix_shuf(x)
        # x = F.interpolate(x, skip.shape[-2:], mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.resb(x)


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
        self.to_qkv = spectral_norm(nn.Conv2d(n_channels, sum(self.qkv_c), kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):       # [B, C, H, W]
        size = x.size()
        qkv = self.to_qkv(x)
        q, k, v = qkv.flatten(2).split(self.qkv_c, dim=1)   # [B, (dq,dk,dv), H*W]
        attn = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)  # [B, lq, lk]
        o = torch.bmm(v, attn)
        del attn, q, k, v, qkv
        o = o.view(*size)  # .contiguous()
        o = o.mul_(self.gamma) + x
        return o


class Unet(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=32, num_scale=1, groups=32, reduction=16, spectral=True,
                 self_attn=False, norm_lyr=nn.InstanceNorm2d):
        super().__init__()
        self.conv_in = ConvNorm(in_c, ngf, ks=3, norm_lyr=norm_lyr, act_cls=nn.ReLU)
        kwargs = dict(groups=groups, reduction=reduction, spectral=spectral, norm_lyr=norm_lyr)
        self.down = self.get_block(ngf, 64, num=1, **kwargs)
        self.down0 = self.get_block(64, 96, num=1, **kwargs)
        self.down1 = self.get_block(96, 128, num=1, self_attn=self_attn, **kwargs)
        self.down2 = self.get_block(128, 256, num=1, **kwargs)
        self.down3 = self.get_block(256, 512, num=1, **kwargs)

        self.middle_conv = nn.Sequential()  # ConvNorm(512, 1024, spectral=spectral, norm_lyr=norm_lyr,
        #                                           act_cls=nn.ReLU),
        #                                  ConvNorm(1024, 512, spectral=spectral, norm_lyr=norm_lyr,
        #                                           act_cls=nn.ReLU),
        #                                 )

        self.up3 = UnetBlock(512, 256, 256, **kwargs)
        self.up2 = UnetBlock(256, 128, 128, **kwargs)
        self.up1 = UnetBlock(128, 96, 96, **kwargs)
        self.up0 = UnetBlock(96, 64, 64, **kwargs)
        self.up = UnetBlock(64, ngf, ngf, **kwargs)

        n_up = (ngf, 64, 96, 128, 256, 512)
        self.deep_convs = nn.ModuleList([nn.Conv2d(n_up[i], out_c, kernel_size=3 if i == 0 else 1,
                                                   padding='same') for i in range(num_scale)])

    def forward(self, x, get_feat=False, encode_only=False):  # 3, 768
        x = self.conv_in(x)        # 32, 768
        d = self.down(x)           # 64, 384
        d0 = self.down0(d)          # 96, 192
        d1 = self.down1(d0)         # 128, 96
        d2 = self.down2(d1)         # 256, 48
        d3 = self.down3(d2)         # 512, 24

        u3 = self.middle_conv(d3)   # 512, 24

        u2 = self.up3(u3, d2)       # 256, 48
        u1 = self.up2(u2, d1)       # 128, 96
        u0 = self.up1(u1, d0)       # 96, 192
        u = self.up0(u0, d)        # 64, 384
        o = self.up(u, x)        # 32, 768
        if get_feat:
            feats = d, d0, d1, d2, d3
            if encode_only:
                return feats
            else:
                return torch.tanh(self.deep_convs[0](o)), feats
        return torch.tanh(self.deep_convs[0](o))

    def get_block(self, ni, nf, num=2, self_attn=False, **kwargs):
        return nn.Sequential(*[ResBlock(ni if i == 0 else nf, nf, stride=2 if i == 0 else 1,
                                        self_attn=self_attn if i == 0 else False, **kwargs)
                               for i in range(num)])


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
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
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


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_lyr=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, gauss_std=0.1):
        super().__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[
            GaussianNoise(gauss_std),
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                GaussianNoise(gauss_std),
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_lyr(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            GaussianNoise(gauss_std),
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_lyr(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[GaussianNoise(gauss_std), nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, inp):
        if self.getIntermFeat:
            res = [inp]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(inp)


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
        if isinstance(inp[0], list):
            loss = 0
            for input_i in inp:
                pred = input_i[-1]
                target_tensor = torch.empty_like(pred).fill_(target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = torch.empty_like(inp[-1]).fill_(target_is_real)
            return self.loss(inp[-1], target_tensor)


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


class VGGLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer_ids = [22, 32, 42]
        self.weights = [5, 15, 2]
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


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
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
