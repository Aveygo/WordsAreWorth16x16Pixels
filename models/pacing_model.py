# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch, numpy as np, os
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

PAD_TOKEN = 50256

# Constants that should not be changed due to some hard-coded values in the original ConvNext-base model
VOCAB_SIZE, N_EMDB, N_POSITIONS = 50257, 768, 1024

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """ Generator Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Generator(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 20, 3], dims=[128, 256, 512, 1024], drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        
        # Stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList() 
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.wte = nn.Embedding(VOCAB_SIZE, N_EMDB)
        self.wpe = nn.Embedding(N_POSITIONS, N_EMDB)

        # Final norm layer (not used in this implementation)
        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        
        self.pre_head = nn.ConvTranspose1d(dims[-1], 1024, kernel_size=12, stride=12)
        
        self.head = nn.Linear(self.wte.weight.shape[1], self.wte.weight.shape[0], bias=False)
        self.head.weight = self.wte.weight

        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def pad_indices(self, indices):

        if indices.shape[1] < N_POSITIONS:
            indices = torch.nn.functional.pad(indices, (0, N_POSITIONS - indices.shape[1]), value=50256)
        else:
            indices = indices[-N_POSITIONS:]
        return indices
    
    def build_image(self, patches):
        # patches: (B, 1024, 3, 16, 16)
        image = torch.zeros((patches.shape[0], 3, 256, 256)).cuda()
        for i in range(16):
            for j in range(16):
                image[:, :, i*16:(i+1)*16, j*16:(j+1)*16] = patches[:, i*16+j, :, :, :]

        return image

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # Check if bias is present (ignore final logit output layer)
            if m.bias is not None:
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # global average pooling, (N, C, H, W) -> (N, C) (not used in this implementation)
        # return self.norm(x.mean([-2, -1]))
        
        # [B, 1024, 8, 8] -> [B, 1024, 64]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.pre_head(x)
        return x

    def forward(self, input_ids):
        # Reverse the order of the tokens
        input_ids = torch.flip(input_ids, [1])

        # Padd with 50256
        input_ids = self.pad_indices(input_ids)

        # Prepare the position ids / embeddings
        position_ids = torch.arange(0, input_ids.size(-1) + 0, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        # Embeddings and position encoding
        x = self.wte(input_ids) + self.wpe(position_ids)

        # Reshape to (B, 1024, 3, 16, 16)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 3, 16, 16))
        
        # Build an image from the patches
        x = self.build_image(x)

        # Run though the convnet
        x = self.forward_features(x)

        # Output logits
        x = self.head(x)

        return x[:, -1]