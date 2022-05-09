import torch
import torch.nn as nn

from einops import rearrange
from torch.nn.modules import activation
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F

from .utils import padding, unpadding

import mmcv
import numpy as np


def conv_1x1_bn(inp, oup,activation="silu"):
    model =  nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )
    if activation =="silu":
      model.append(nn.SiLU())
    elif activation=="relu":
      model.append(nn.ReLU())
    return model


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1,dilation=None,activation="silu",padding=1):
    if dilation:
      model = nn.Sequential(
          nn.Conv2d(inp, oup, kernal_size, stride, padding, bias=False,dilation=dilation),
          nn.BatchNorm2d(oup),
      )
    else:
      model = nn.Sequential(
          nn.Conv2d(inp, oup, kernal_size, stride, padding, bias=False),
          nn.BatchNorm2d(oup),
      )
    if activation=="silu":
      model.append(nn.SiLU())
    elif activation =="relu":
      model.append(nn.ReLU())
    return model


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)

        self.attend = nn.Softmax(dim = -1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim,bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4,kernel_size=1,print_flag=False):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if print_flag:
              hidden_dim=inp
              second_hidden_dim = 64
              second_kernel_size = 1
              groups = 1
              print(hidden_dim)
            else:
              groups = hidden_dim
              second_kernel_size=3
              second_hidden_dim = hidden_dim
              #print(nn.Conv2d(inp, inp, 1, 1, 0, bias=False))
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, kernel_size, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, second_hidden_dim, second_kernel_size, stride, 1, groups=groups, bias=False),
                nn.BatchNorm2d(second_hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        
        self.new_conv2 = nn.Conv2d(channel, dim, 1, 1, 0, bias=False)

        #self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.prenorm = LayerNorm(dim)#TODO: check how to use
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.new_conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = self.prenorm(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2), n_cls=150, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        print(f'MODEL ARGS: {channels}')

        self.patch_size = patch_size[0]
        self.n_cls = n_cls

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]
        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)
        
        self.mv1 = MV2Block(channels[0], channels[1], 1, expansion)
        self.mv2 = MV2Block(channels[1], channels[2], 2, expansion)
        self.mv3 = MV2Block(channels[2], channels[3], 1, expansion)
        self.mv4 = MV2Block(channels[2], channels[3], 1, expansion)  # Repeat
        self.mv5 = MV2Block(channels[3], channels[4], 2, expansion)
        
        self.mvit1 = MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2))


        self.mv6 = MV2Block(channels[5], channels[6], 2, expansion) #,kernel_size=3,print_flag=True))
        self.mvit2 = MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*2))
       
        self.mv7 = MV2Block(channels[7], channels[8], 1, expansion)
        
        #self.mvit = nn.ModuleList([])
        self.mvit3 = MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*2))
        #this layer is actually not used...
        #self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        # self.pool = nn.AvgPool2d(ih//32, 1)
        # self.fc = nn.Linear(channels[-1], num_classes, bias=False)

        #Deeplab part:

        decoder_channels = channels[-2]
        self.dl1 = conv_1x1_bn(decoder_channels, 256, activation="relu")
        self.dl2 = conv_nxn_bn(decoder_channels, 256, kernal_size=3, activation="relu", dilation=6, padding=6)
        self.dl3 = conv_nxn_bn(decoder_channels, 256, kernal_size=3, activation="relu", dilation=12, padding=12)
        self.dl4 = conv_nxn_bn(decoder_channels, 256, kernal_size=3, activation="relu", dilation=18, padding=18)
        self.dl5 = nn.AdaptiveAvgPool2d(1)
        self.dl6 = conv_1x1_bn(decoder_channels, 256,activation="relu")
        self.dl7 = conv_1x1_bn(1280, 256,activation="relu")
        self.dldrop = nn.Dropout(p=0.1, inplace=False)

        output_classes = self.n_cls
        self.dlclassifier1 = nn.Dropout2d(p=0.1, inplace=False)
        self.dlclassifier2 = nn.Conv2d(256,output_classes,1,1)
        self.upsampled = nn.Upsample(scale_factor=16,mode="bilinear")

        self.init_weights(pretrained_path)

    def forward(self, x, return_features=False):
        H_ori, W_ori = x.size(2), x.size(3)
        x = padding(x, self.patch_size)
        H, W = x.size(2), x.size(3)

        import copy 
        outputs = []
        x = self.conv1(x)

        x = self.mv1(x)
        outputs.append(copy.copy(x))

        x = self.mv2(x)
        x = self.mv3(x)
        x = self.mv4(x)      # Repeat
        outputs.append(copy.copy(x))

        x = self.mv5(x)
        x = self.mvit1(x)
        outputs.append(copy.copy(x))

        x = self.mv6(x)
        x = self.mvit2(x)
        outputs.append(copy.copy(x))

        x = self.mv7(x)
        x = self.mvit3(x)
        outputs.append(copy.copy(x))

        #pass through ASPP:
        aspps = []
        aspps.append(self.dl1(x))
        aspps.append(self.dl2(x))
        aspps.append(self.dl3(x))
        aspps.append(self.dl4(x))
        x_size = x.shape[-2:]
        pooled = self.dl5(x)
        pooled = self.dl6(pooled)

        aspps.append(nn.functional.interpolate(pooled, size=x_size, mode="bilinear", align_corners=False))
        aspps = torch.cat(aspps, dim=1)
        #x = self.dl6()
        x = self.dl7(aspps)
        x = self.dldrop(x)
        
        #classify
        x = self.dlclassifier1(x)
        x = self.dlclassifier2(x)
        x = self.upsampled(x)

        x = F.interpolate(x, size=(H, W), mode="bilinear")
        x = unpadding(x, (H_ori, W_ori))

        #x = self.conv2(x)

        # x = self.pool(x).view(-1, x.shape[1])
        # x = self.fc(x)
        return x

    def init_weights(self, pretrained_path=None):
        print(f'IN INIT WEIGHTS: {pretrained_path}')
        model_loaded = torch.load(pretrained_path)
        loaded_weights = list(model_loaded.values())
        self_model = self.state_dict()

        assert len(self_model) == len(loaded_weights)

        for i, key in enumerate(self_model):
          if key.startswith("dl1"):
              print(f'breaking at key {key}')
              break

          self_model[key] = loaded_weights[i]
        self.load_state_dict(self_model)
        print("ITS ALIVE")


    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.n_cls), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.n_cls)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            print('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 256]
    return MobileViT((512, 512), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)

