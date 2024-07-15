## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import os

logger = logging.getLogger(__name__)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, conf_weights=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if conf_weights is not None:
            attn = attn * conf_weights.unsqueeze(1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, conf_weights=None):
        
        if conf_weights is not None:
            attn_output = self.attn(self.norm1(x), conf_weights)
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MultiView_MPL(nn.Module):
    def __init__(self, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_views=5,
                 add_confidence_input=False,
                 mult_confidence_emb=False,
                 concat_confidence_emb=False,
                 confidence_input_as_third=False,
                #  drop_emb_conf_rate=False,
                 pose_3d_emb_learnable=False,
                 linear_weighted_mean=False,
                 pos_embedding_type="learnable",
                 add_3D_pos_encoding_in_Spatial=False,
                 input_rays_as_token=False,
                 add_3D_pos_encoding_to_rays=False,
                 confidence_as_attention_uncertainty_weight=False,
                 multiple_spatial_blocks=False,
                 no_transformer_spt=False,
                 no_transformer_fpt=False,
                 confidence_in_FPT=False,
                 deep_head=False,
                 head_kadkhod=False,
                 hidden_dim=1024,
                 FPT_blocks_view_keypoint_tokens=False,):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        # num_views = 5
        self.num_joints = num_joints
        self.num_views = num_views
        self.embed_dim_ratio = embed_dim_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        if input_rays_as_token:
            embed_dim = embed_dim_ratio * 2 * num_joints   # because we add ray tokens
        out_dim = num_joints * 3     #### output dimension is num_joints * 3
        
        # assert pos_embedding_type in ['sine','none','learnable','sine-full']
        # self._make_position_embedding(embed_dim_ratio, pos_embedding_type)
        
        self.deep_head = deep_head
        self.head_kadkhod = head_kadkhod
        self.hidden_dim = hidden_dim
        self.confidence_input_as_third = confidence_input_as_third
        self.multiple_spatial_blocks = multiple_spatial_blocks
        self.no_transformer_spt = no_transformer_spt
        self.no_transformer_fpt = no_transformer_fpt
        
        self.FPT_blocks_view_keypoint_tokens = FPT_blocks_view_keypoint_tokens
        
        ### spatial patch embedding
        if self.confidence_input_as_third:
            if self.multiple_spatial_blocks:
                self.Spatial_patch_to_embedding = nn.ModuleList([nn.Linear(in_chans + 1, embed_dim_ratio) for _ in range(num_views)])
            else:
                self.Spatial_patch_to_embedding = nn.Linear(in_chans + 1, embed_dim_ratio)
        else:
            if self.multiple_spatial_blocks:
                self.Spatial_patch_to_embedding = nn.ModuleList([nn.Linear(in_chans, embed_dim_ratio) for _ in range(num_views)])
            else:
                self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        
        self.add_confidence_input = add_confidence_input
        self.mult_confidence_emb = mult_confidence_emb
        self.concat_confidence_emb = concat_confidence_emb
        if self.concat_confidence_emb:
            self.add_confidence_input = False
            self.mult_confidence_emb = False
            self.concat_confidence_emb = False
        # self.drop_emb_conf_rate = drop_emb_conf_rate
        
        self.confidence_to_embedding = None
        if self.add_confidence_input or self.mult_confidence_emb or self.concat_confidence_emb:
            if self.multiple_spatial_blocks:
                self.confidence_to_embedding = nn.ModuleList([nn.Linear(1, embed_dim_ratio) for _ in range(num_views)])
            else:
                self.confidence_to_embedding = nn.Linear(1, embed_dim_ratio)
        
        self.confidence_as_attention_uncertainty_weight = confidence_as_attention_uncertainty_weight
        
        if self.concat_confidence_emb:
            embed_dim_ratio = embed_dim_ratio * 2
            embed_dim = embed_dim_ratio * num_joints 
            
        if self.multiple_spatial_blocks:
            self.Spatial_pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio)) for _ in range(num_views)])
        else:
            self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.pos_3d_linear = nn.Linear(3, embed_dim_ratio)
        
        # >>>>>>>>>>>>>>>>>>>>>>> learnable 3D embedding >>>>>>>>>>>>>>>>>>>>>>>>>
        self.pose_3d_emb_learnable = pose_3d_emb_learnable
        # self.pos_3d_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.add_3D_pos_encoding_in_Spatial = add_3D_pos_encoding_in_Spatial
        
        # self.pos_3d_view_coding = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.add_3D_pos_encoding_to_rays = add_3D_pos_encoding_to_rays
        if self.add_3D_pos_encoding_to_rays:
            if add_3D_pos_encoding_in_Spatial:
                self.pos_3d_linear = nn.Linear(3, embed_dim_ratio)
            else:    
                self.pos_3d_linear = nn.Linear(3, embed_dim_ratio * 2)
            self.pos_3d_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio * 2))
            self.pos_3d_view_coding = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio * 2))
        else:
            self.pos_3d_linear = nn.Linear(3, embed_dim_ratio)
            self.pos_3d_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
            self.pos_3d_view_coding = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        
        # >>>>>>>>>>>>>>>>>>>>>>> ray token >>>>>>>>>>>>>>>>>>>>>>>>>
        self.add_3D_pos_encoding_to_rays = add_3D_pos_encoding_to_rays
        self.input_rays_as_token = input_rays_as_token
        if self.input_rays_as_token:
            self.ray_to_embedding = nn.Linear(3, embed_dim_ratio)
            
        self.confidence_in_FPT = confidence_in_FPT
        if self.confidence_in_FPT:
            self.confidence_to_embedding_FPT = nn.Linear(1, embed_dim_ratio)
            


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        ##### create SPT blocks
        if self.multiple_spatial_blocks:
            self.Spatial_blocks = nn.ModuleList([
                    nn.ModuleList([
                        Block(
                            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                        for i in range(depth)]) for _ in range(num_views)
            ])
        else:
            self.Spatial_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
            
        if self.no_transformer_spt:
            self.Spatial_blocks = nn.ModuleList([])
        
        ##### create FPT blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        if self.FPT_blocks_view_keypoint_tokens:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
        
        if self.no_transformer_fpt:
            self.blocks = nn.ModuleList([])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        if input_rays_as_token:
            embed_dim = embed_dim // 2
        self.View_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.linear_weighted_mean = linear_weighted_mean
        if self.linear_weighted_mean:
            self.weighted_mean = nn.Linear(num_views * embed_dim, embed_dim)
        else:
            self.weighted_mean = torch.nn.Conv1d(in_channels=num_views, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
        if self.deep_head:
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim , self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim , self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim , self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim , out_dim),
            )
        if self.head_kadkhod:
            self.head = nn.ModuleList([
                nn.Sequential(nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            nn.Linear(self.hidden_dim, out_dim)),

                nn.Sequential(nn.Sequential(nn.Linear(out_dim + embed_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            torch.nn.Linear(self.hidden_dim, out_dim)),

                nn.Sequential(nn.Sequential(nn.Linear(out_dim + embed_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(True)),
                            torch.nn.Linear(self.hidden_dim, out_dim))
            ])
            
        
        
    # def _make_position_embedding(self, d_model, pe_type='sine'):
    #     '''
    #     d_model: embedding size in transformer encoder
    #     '''
    #     assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
    #     if pe_type == 'none':
    #         self.pos_embedding = None
    #         print("==> Without any PositionEmbedding~")
    #     else:
    #         if pe_type == 'learnable':
    #             self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_joints, d_model))
    #             trunc_normal_(self.pos_embedding, std=.02)
    #             print("==> Add Learnable PositionEmbedding~")
    #         else:
    #             self.pos_embedding = nn.Parameter(
    #                 self._make_sine_position_embedding(d_model),
    #                 requires_grad=False)
    #             print("==> Add Sine PositionEmbedding~")
                
    # def _make_sine_position_embedding(self, d_model, temperature=10000,
    #                                   scale=2 * math.pi):
        
    #     position = torch.arange(0, self.num_joints, dtype=torch.float32).unsqueeze(1).float()
    #     div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(temperature) / d_model))
    #     pos_enc = torch.sin(position * div_term)
    #     return pos_enc


    def Spatial_forward_features(self, x, ray=None, center=None, view_number=None):
        # b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        # x = rearrange(x, 'b c f p  -> (b f) p  c', )
        if self.confidence_as_attention_uncertainty_weight:
            attention_weights = x[:, :, 2:3].clone()
            
        if self.multiple_spatial_blocks:
            spt_joint_embed = self.Spatial_patch_to_embedding[view_number]
        else:
            spt_joint_embed = self.Spatial_patch_to_embedding
        if self.confidence_input_as_third:
            # x_embedded = self.Spatial_patch_to_embedding(x[:, :, 0:3])
            x_embedded = spt_joint_embed(x[:, :, 0:3])
        else:
            # x_embedded = self.Spatial_patch_to_embedding(x[:, :, 0:2])
            x_embedded = spt_joint_embed(x[:, :, 0:2])
            
        if self.multiple_spatial_blocks and self.confidence_to_embedding is not None:
            spt_conf_embed = self.confidence_to_embedding[view_number]
        else:
            spt_conf_embed = self.confidence_to_embedding
            
        if self.add_confidence_input:
            conf = spt_conf_embed(x[:, :, 2:3])
            x_embedded += conf
        if self.mult_confidence_emb:
            conf = spt_conf_embed(x[:, :, 2:3])
            x_embedded *= conf
        if self.concat_confidence_emb:
            conf = spt_conf_embed(x[:, :, 2:3])
            x_embedded = torch.cat([x_embedded, conf], dim=2)
        # if self.drop_emb_conf_rate:
        #     x_embedded = F.dropout(x_embedded, p=1 - x[:, :, 2:3], training=self.training)
        if self.multiple_spatial_blocks:
            x_embedded += self.Spatial_pos_embed[view_number]
        else:
            x_embedded += self.Spatial_pos_embed
        x = x_embedded
        
        # >>>>>>>>>>>>>>>> 3D position encoding >>>>>>>>>>>>>>>>
        if self.add_3D_pos_encoding_in_Spatial and ray is not None and center is not None:
            b = x.shape[0]
            if self.pose_3d_emb_learnable:
                pos_emb_3d = self.pos_3d_embed.expand(b, -1, -1)
            else:
                vec_c_p = F.normalize(ray - center, dim=2, p=2)     # (B, HW, 3)
                pos_emb_3d = self.pos_3d_linear(vec_c_p)   
            x += pos_emb_3d
        
        x = self.pos_drop(x)

        if self.multiple_spatial_blocks:
            spt = self.Spatial_blocks[view_number]
        else:
            spt = self.Spatial_blocks
        # for ix, blk in enumerate(self.Spatial_blocks):
        for ix, blk in enumerate(spt):
            if self.confidence_as_attention_uncertainty_weight:
                x = blk(x, attention_weights)
            if ix == len(spt) - 1:
                x = blk(x)
            x = blk(x)

        x = self.Spatial_norm(x)
        # x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b  = x.shape[0]
        # x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for ix, blk in enumerate(self.blocks):
            if ix == len(self.blocks) - 1:
                x = blk(x)
            x = blk(x)
            
        if self.input_rays_as_token and not self.add_3D_pos_encoding_to_rays:
            b, _, _ = x.shape
            x = x.view(b, self.num_views, 2, self.num_joints, self.embed_dim_ratio)
            x = x[:,:,0,:,:]  # remove the ray tokens
            x = x.view(b, self.num_views, -1)
        elif self.add_3D_pos_encoding_to_rays:
            b, _, _ = x.shape
            x = x.view(b, self.num_views, self.num_joints, self.embed_dim_ratio * 2)
            x = x[:,:,:,:self.embed_dim_ratio]  # remove the ray tokens
            x = x.reshape(b, self.num_views, -1)
            
        if self.FPT_blocks_view_keypoint_tokens:
            x = x.view(b, self.num_views, -1)

        x = self.View_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        if self.linear_weighted_mean:
            x = x.view(b, -1)
            x = self.weighted_mean(x)
        else:
            x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x


    def forward(self, poses, rays=None, centers=None):
        # x = x.permute(0, 3, 1, 2)
        # b, _, _, p = x.shape
        
        b = poses[0].shape[0]
        xs = []
        num_views = len(poses)
        # pos_embs_3d = []
        for i, pose in enumerate(poses):
            # b, n, _ = pose.shape
            # # >>>>>>>>>>>>>>>> 2D position encoding >>>>>>>>>>>>>>>>
            # pos_emb_2d = self.pos_embedding[:, :n].expand(b, -1, -1)    # (B, HW, C) only consider sine-full 2D PE here
            # pose += pos_emb_2d
            x = self.Spatial_forward_features(pose, ray=rays[i], center=centers[i], view_number=i)
            
            if self.confidence_in_FPT:
                conf = self.confidence_to_embedding_FPT(pose[:, :, 2:3])
                x += conf
            
            if self.add_3D_pos_encoding_to_rays and self.input_rays_as_token:
                ray_emb = self.ray_to_embedding(rays[i] - centers[i])
                x = torch.cat([x, ray_emb], dim=2)
                
            # >>>>>>>>>>>>>>>> 3D position encoding >>>>>>>>>>>>>>>>
            if not self.add_3D_pos_encoding_in_Spatial:
                if self.pose_3d_emb_learnable:
                    pos_emb_3d = self.pos_3d_embed.expand(b, -1, -1)
                else:
                    vec_c_p = F.normalize(rays[i] - centers[i], dim=2, p=2)     # (B, HW, 3)
                    pos_emb_3d = self.pos_3d_linear(vec_c_p)                    # (B, HW, 3)
            else:
                pos_emb_3d = self.pos_3d_view_coding.expand(b, -1, -1)
            
            x += pos_emb_3d
            
            # >>>>>>>>>>>>>>>> ray token >>>>>>>>>>>>>>>>
            if not self.add_3D_pos_encoding_to_rays and self.input_rays_as_token:
                ray_emb = self.ray_to_embedding(rays[i] - centers[i])
            
                x = torch.cat([x, ray_emb], dim=1)
                
            x = x.view(b, -1)
            xs.append(x)
            # pos_embs_3d.append(pos_emb_3d)
            
        xs = torch.cat(xs, dim=1)
        if self.FPT_blocks_view_keypoint_tokens:
            xs = xs.view(b, num_views * self.num_joints, -1)
        else:
            xs = xs.view(b, num_views, -1)
        
        # b, _, _ = xs.shape
        # xs = xs.view(b, -1)
        # pos_embs_3d = torch.cat(pos_embs_3d, dim=1)
            
        x = self.forward_features(xs)
        if self.head_kadkhod:
            x = x.view(b, -1)
            x_intermediate = []
            x1 = self.head[0](x)
            x_intermediate.append(x1.view(b, -1, 3))
            x2 = self.head[1](torch.cat([x1, x], dim=1))
            x_intermediate.append(x2.view(b, -1, 3))
            x3 = self.head[2](torch.cat([x2, x], dim=1))
            x = x3.view(b, -1, 3)
                
            return x, x_intermediate
        elif self.deep_head:
            x = x.view(b, -1)
            x = self.head(x)
        else:
            x = self.head(x)

        x = x.view(b, -1, 3)

        return x


class MultiView_MPL_G(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(MultiView_MPL_G, self).__init__()

        print(cfg.NETWORK)
        # num_views = 5 if cfg.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') else 4
        if cfg.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') or cfg.DATASET.TEST_DATASET.startswith('multiview_amass_cmu_panoptic_mpl'):
            num_views = 5
        else:
            num_views = 4
            
        if cfg.DATASET.TRAIN_VIEWS is not None:
            num_views = len(cfg.DATASET.TRAIN_VIEWS)
            if cfg.DATASET.USE_HELPER_CAMERAS:
                assert cfg.DATASET.TRAIN_VIEWS_HELPER is not None
                num_views += len(cfg.DATASET.TRAIN_VIEWS_HELPER)
                
        if cfg.DATASET.TRAIN_ON_ALL_CAMERAS and cfg.DATASET.TEST_ON_ALL_CAMERAS:
            num_views = cfg.DATASET.N_VIEWS_TRAIN_TEST_ALL
                
            
        self.init_weights_from = cfg.NETWORK.INIT_WEIGHTS_FROM

        ##################################################
        self.features = MultiView_MPL(
                                 num_joints = cfg.NETWORK.NUM_JOINTS,
                                 embed_dim_ratio=cfg.NETWORK.DIM,
                                 depth=cfg.NETWORK.TRANSFORMER_DEPTH,
                                 num_heads=cfg.NETWORK.TRANSFORMER_HEADS,
                                 drop_rate=cfg.NETWORK.TRANSFORMER_DROP_RATE,
                                 attn_drop_rate=cfg.NETWORK.TRANSFORMER_ATTN_DROP_RATE,
                                 drop_path_rate=cfg.NETWORK.TRANSFORMER_DROP_PATH_RATE,
                                 num_views=num_views,
                                 add_confidence_input=cfg.NETWORK.TRANSFORMER_ADD_CONFIDENCE_INPUT,
                                 mult_confidence_emb=cfg.NETWORK.TRANSFORMER_MULT_CONFIDENCE_EMB,
                                 concat_confidence_emb=cfg.NETWORK.TRANSFORMER_CONCAT_CONFIDENCE_EMB,
                                 confidence_input_as_third=cfg.NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD,
                                #  drop_emb_conf_rate=cfg.NETWORK.TRANSFORMER_DROP_EMB_CONF_RATE,
                                 pose_3d_emb_learnable=cfg.NETWORK.POSE_3D_EMB_LEARNABLE,
                                 linear_weighted_mean=cfg.NETWORK.TRANSFORMER_LINEAR_WEIGHTED_MEAN,
                                 add_3D_pos_encoding_in_Spatial=cfg.NETWORK.TRANSFORMER_ADD_3D_POS_ENCODING_IN_SPATIAL,
                                 input_rays_as_token=cfg.NETWORK.TRANSFORMER_INPUT_RAYS_AS_TOKEN,
                                 add_3D_pos_encoding_to_rays=cfg.NETWORK.TRANSFORMER_ADD_3D_POS_ENCODING_TO_RAYS,
                                 confidence_as_attention_uncertainty_weight=cfg.NETWORK.TRANSFORMER_CONF_ATTENTION_UNCERTAINTY_WEIGHT,
                                 multiple_spatial_blocks=cfg.NETWORK.TRANSFORMER_MULTIPLE_SPATIAL_BLOCKS,
                                 no_transformer_spt=cfg.NETWORK.TRANSFORMER_NO_SPT,
                                 no_transformer_fpt=cfg.NETWORK.TRANSFORMER_NO_FPT,
                                 confidence_in_FPT=cfg.NETWORK.TRANSFORMER_CONFIDENCE_IN_FPT,
                                 deep_head=cfg.NETWORK.TRANSFORMER_OUTPUT_HEAD_DEEP,
                                 head_kadkhod=cfg.NETWORK.TRANSFORMER_OUTPUT_HEAD_KADKHOD,
                                 hidden_dim=cfg.NETWORK.TRANSFORMER_OUTPUT_HEAD_HIDDEN_DIM,
                                 FPT_blocks_view_keypoint_tokens=cfg.NETWORK.TRANSFORMER_FPT_BLOCKS_VIEW_KEYPOINT_TOKENS,
                                 )
        ###################################################3

    def forward(self, x, centers=None, rays=None):
        x = self.features(x, rays=rays, centers=centers,)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            if 'multiview_h36m' in pretrained or 'multiview_amass_h36m' in pretrained or 'multiview_cmu_panoptic' in pretrained or 'multiview_amass_cmu_panoptic_mpl' in pretrained:
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> from H36M pretrained >>>>>>>>>>>>>>>>>>>>>>>>>>>
                logger.info('=> loading Pretrained model {}'.format(pretrained))
                pretrained_state_dict = torch.load(pretrained, map_location='cpu')
                self.load_state_dict(pretrained_state_dict, strict=False)
            else:
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> from COCO pretrained >>>>>>>>>>>>>>>>>>>>>>>>>>>
                logger.info('=> init final MLP head from normal distribution')
                for m in self.features.mlp_head.modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                pretrained_state_dict = torch.load(pretrained, map_location='cpu')
                logger.info('=> loading COCO Pretrained model {}'.format(pretrained))
                existing_state_dict = {}
                for name, m in pretrained_state_dict.items():
                    if name in self.state_dict():
                        #if 'mlp_head' in name or 'pos_embedding' in name or 'keypoint_token' in name or 'patch_to_embedding' in name:       # 2D Pos Embeddings
                        #    continue
                        if 'keypoint_token' in name:
                            new_m = torch.zeros(1, 17, 192)
                            # Human 36M -> MPII
                            # map_idx = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, 13, 14, 15, 12, 11, 10]
                            # Human 36M -> COCO
                            map_idx = [12, 12, 14, 16, 11, 13, 15, 11, 1, 0, 2, 5, 7, 9, 6, 8, 10]
                            new_m[0] = m[0][map_idx]
                            m = new_m
                            print('Shift Token ...')

                        existing_state_dict[name] = m
                        logger.info(":: {} is loaded from {}".format(name, pretrained))
                        print('Size: ', m.shape)

                self.load_state_dict(existing_state_dict, strict=False)

        elif self.init_weights_from == 'xavier_uniform':
            logger.info('=> init weights from xavier uniform distribution')
            for m in self.modules():
                if not isinstance(m, MultiView_MPL_G) or not isinstance(m, MultiView_MPL_G):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> from scratch >>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


def get_multiview_mpl_net(cfg, is_train, **kwargs):
    model = MultiView_MPL_G(cfg, **kwargs)
    if is_train and cfg.NETWORK.INIT_WEIGHTS:
        model.init_weights(cfg.NETWORK.PRETRAINED)

    return model