import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

from .layers import conv_layer, deconv_layer
import os
from functools import partial
from detectron2.layers import ShapeSpec

class ClipOutput(dict):
    def __init__(self, spacial_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacial_shape = spacial_shape

    def save(self, idx: int, clip_feat: torch.Tensor, cls_token):
        l, n, c = clip_feat.shape
        self[idx] = (
            clip_feat.permute(1, 2, 0).reshape(n, c, *self.spacial_shape)
        )  # n, c, h, w
        self[f"{idx}_cls_token"] = cls_token  # 1, n, c

class Intertwiner_ViT(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 use_side = False,
                 use_txt_add = False,
                 shared_weights = False,
                ):
        super().__init__()
        
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers

        self.n_ctx_visual = 0
        # visual_ctx_vectors = torch.empty(self.n_ctx_visual, 768)
        # nn.init.normal_(visual_ctx_vectors, std=0.02)
        # self.visual_propogator = nn.Parameter(visual_ctx_vectors)

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        self.proj  = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 768)
        )
        if use_txt_add:
            self.proj_txt = nn.Sequential(
                nn.Linear(768, 32),
                nn.ReLU(),
                nn.Linear(32, 512)
            )
        self.use_txt_add = use_txt_add
        self.use_side = use_side
        self.initialize_parameters()
        self.output_shapes = self.output_shapes()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, img, text, backbone):
        B=img.shape[0]
        # vision
        img = img.type(backbone.dtype)
        vis_enc = backbone.visual
        vis = vis_enc.conv1(img)  # shape = [*, width, grid, grid]
        b,c,h,w=vis.shape

        vis = vis.reshape(vis.shape[0], vis.shape[1], -1)  # shape = [*, width, grid ** 2]
        vis = vis.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        vis = torch.cat([
            vis_enc.class_embedding.to(vis.dtype) + torch.zeros(vis.shape[0], 1, vis.shape[-1], 
                dtype=vis.dtype, device=vis.device), vis], dim=1)  # shape = [*, grid ** 2 + 1, width]
        vis = vis + vis_enc.positional_embedding.to(vis.dtype)
        vis = vis_enc.ln_pre(vis)
        
        # language
        txt = backbone.token_embedding(text).type(
            backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = backbone.transformer
        txt = txt + backbone.positional_embedding.type(backbone.dtype)[:txt.size(1)]
        

        # fusion
        stage_i = 0
        vis_outs = []

        txt_shape = txt.shape[1]
        if self.use_txt_add:
            txt_add = self.proj_txt(vis[:,0:1,:])
        vis= torch.cat((
                vis[:, :1, :],
                # self.visual_propogator.expand(B, -1, -1),
                self.proj(txt),
                vis[:, 1:, :]
            ), dim=1)
        if self.use_txt_add:
            txt = torch.cat([
                txt_add,
                txt
            ],dim=1)

        vis = vis.permute(1, 0, 2)  # NLD -> LND
        txt = txt.permute(1, 0, 2)  # NLD -> LND

        # output for GST
        if self.use_side:
            outputs = ClipOutput(spacial_shape=(h, w))
            outputs.save(0, vis[1+self.n_ctx_visual+txt_shape:, :, :],vis[0:1, :, :])

        for i in range(self.num_layers):
            vis = vis_enc.transformer.resblocks[i](vis)
            txt = txt_enc.resblocks[i](txt)
            if self.use_side:
                outputs.save(i, vis[1+self.n_ctx_visual+txt_shape:, :, :],vis[0:1, :, :])
            if (i+1)%4 == 0:          
                stage_i += 1
                if stage_i < self.num_stages:
                    vis_out = vis[1+self.n_ctx_visual+txt_shape:, :, :].permute(1, 2, 0) # B, D, N
                    B, C, N = vis_out.shape
                    H = int(N ** 0.5)
                    W = N // H
                    vis_out = vis_out.reshape(B, C, H, W) # B, D, H, W
                    vis_outs.append(vis_out)  

        # After fusion
        # vision
        # 197, 64, 768 -> 64, 197, 768
        vis = torch.cat([vis[0, :, :].unsqueeze(dim=0),vis[1+self.n_ctx_visual+txt_shape:, :, :]],dim=0)
        vis = vis.permute(1, 0, 2)  # LND -> NLD

        # x = vis_enc.ln_post(x[:, 0, :])
        # 64, 197, 768 -> 64, 196, 768
        vis = vis_enc.ln_post(vis[:, 1:, :])

        if vis_enc.proj is not None:
            vis = vis @ vis_enc.proj

        # 64, 196, 512 -> 64, 512, 196
        B, N, C = vis.shape
        H = int(N ** 0.5)
        W = N // H        
        vis = vis.permute(0, 2, 1).reshape(B, C, H, W) # B, N, D -> B, D, N -> B, D, H, W
        vis_outs.append(vis) 

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        output = vis_outs, txt, state, outputs

        return output

    def output_shapes(self):
        return {
            i: ShapeSpec(channels=768)
            for i in range(self.num_layers + 1)
        }


class Intertwiner_EVA(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 use_side=False,
                 use_txt_add=False,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                ):
        super().__init__()
        
        self.d_img = d_img
        self.use_txt_add = use_txt_add
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.use_side = use_side
        

        self.n_ctx_visual = 0
        # visual_ctx_vectors = torch.empty(self.n_ctx_visual, 768)
        # nn.init.normal_(visual_ctx_vectors, std=0.02)
        # self.visual_propogator = nn.Parameter(visual_ctx_vectors)

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        
        self.proj  = nn.Sequential(
            nn.Linear(self.d_txt, 32),
            nn.ReLU(),
            nn.Linear(32, 768)
        )
        if use_txt_add:
            self.proj_txt = nn.Sequential(
                nn.Linear(768, 32),
                nn.ReLU(),
                nn.Linear(32, 512)
            )
        self.initialize_parameters()
        self.output_shapes = self.output_shapes()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, img, text, backbone):
        b, c, w, h = img.shape
        # vision
        img = img.type(backbone.dtype)
        vis_enc = backbone.visual
        vis = vis_enc.patch_embed(img)
        b, seq_len, c = vis.size()

        cls_tokens = vis_enc.cls_token.expand(b, -1, -1)
        vis = torch.cat((cls_tokens, vis), dim=1)
        
        def resized_pos_embed(in_res, w, h, mode="bicubic"):
            #assert L == (input_resolution // self.patch_size) ** 2 + 1
            _, L, D = vis_enc.pos_embed.shape

            in_side = in_res // vis_enc.patch_size
            #tgt_side = tgt_res // self.patch_size
            cls_pos = vis_enc.pos_embed[:, 0]  # 1 D
            pos_embed = vis_enc.pos_embed[:, 1:].reshape(1, in_side, in_side, D).permute(0, 3, 1, 2) # L-1 D -> 1 D S S
            resized_pos_embed = F.interpolate(pos_embed, size=(w, h), mode=mode, align_corners=False,) # 1 D S S -> 1 D S' S'
            resized_pos_embed = resized_pos_embed.squeeze(0).reshape(D, -1).T # L'-1 D
            return torch.cat((cls_pos, resized_pos_embed), dim=0).unsqueeze(0)

        tgt_side_w = int(w // vis_enc.patch_size)
        tgt_side_h = int(h // vis_enc.patch_size)
        if vis_enc.pos_embed is not None:
            if vis.shape[1] != vis_enc.pos_embed.shape[0]:
                vis = vis + resized_pos_embed(224, tgt_side_w, tgt_side_h).to(vis.dtype)
            else:
                vis = vis + vis_enc.pos_embed
        vis = vis_enc.pos_drop(vis)
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if vis_enc.training and not isinstance(vis_enc.patch_dropout, nn.Identity):
                vis, patch_indices_keep = vis_enc.patch_dropout(vis)
                vis_enc.rope.forward = partial(vis_enc.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                vis_enc.rope.forward = partial(vis_enc.rope.forward, patch_indices_keep=None)
                vis = vis_enc.patch_dropout(vis)
        else:
            vis = vis_enc.patch_dropout(vis)

        rel_pos_bias = vis_enc.rel_pos_bias() if vis_enc.rel_pos_bias is not None else None

        
        # language
        txt_enc = backbone.text
        cast_dtype = txt_enc.transformer.get_cast_dtype()
        txt = txt_enc.token_embedding(text).to(cast_dtype)   # [batch_size, n_ctx, d_model]

        txt = txt + txt_enc.positional_embedding.to(cast_dtype)[:txt.size(1)]
        

        # fusion
        stage_i = 0
        vis_outs = []
        
        txt_shape = txt.shape[1]
        if self.use_txt_add:
            txt_add = self.proj_txt(vis[:,0:1,:])
        vis= torch.cat((
                vis[:, :1, :],
                # self.visual_propogator.expand(B, -1, -1),
                self.proj(txt),
                vis[:, 1:, :]
            ), dim=1)
        if self.use_txt_add:
            txt = torch.cat([
                txt_add,
                txt
            ],dim=1)

        vis = vis.permute(1, 0, 2)  # NLD -> LND
        txt = txt.permute(1, 0, 2)  # NLD -> LND

      

        # output for SAN 
        if self.use_side:
            outputs = ClipOutput(spacial_shape=(tgt_side_h, tgt_side_w))
            outputs.save(0, vis[1+self.n_ctx_visual+txt_shape:, :, :],vis[0:1, :, :])

        for i in range(self.num_layers):
        
            vis = vis_enc.blocks[i](vis, rel_pos_bias=rel_pos_bias)
            txt = txt_enc.transformer.resblocks[i](txt, attn_mask=txt_enc.attn_mask)
            if self.use_side:
                outputs.save(i, vis[1+self.n_ctx_visual+txt_shape:, :, :],vis[0:1, :, :])
            if (i+1)%4 == 0:          
                stage_i += 1
                if stage_i < self.num_stages:
                    vis_out = vis[1+self.n_ctx_visual+txt_shape:, :, :].permute(1, 2, 0) # B, D, N
                    B, C, N = vis_out.shape
                    H = int(N ** 0.5)
                    W = N // H
                    vis_out = vis_out.reshape(B, C, H, W) # B, D, H, W
                    vis_outs.append(vis_out)  

        # After fusion
        # vision
        # 197, 64, 768 -> 64, 197, 768
        vis = torch.cat([vis[0:1, :, :],vis[1+self.n_ctx_visual+txt_shape:, :, :]],dim=0)
        vis = vis.permute(1, 0, 2)  # LND -> NLD

        # x = vis_enc.ln_post(x[:, 0, :])
        # 64, 197, 768 -> 64, 196, 768
        # vis = vis_enc.ln_post(vis[:, 1:, :])

        if vis_enc.head is not None:
            vis = vis_enc.head(vis)
        # 64, 196, 512 -> 64, 512, 196
        # without class token
        vis = vis[:,1:,:]
        B, N, C = vis.shape
        H = int(N ** 0.5)
        W = N // H        
        vis = vis.permute(0, 2, 1).reshape(B, C, H, W) # B, N, D -> B, D, N -> B, D, H, W
        vis_outs.append(vis) 

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = txt_enc.ln_final(txt).type(cast_dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ txt_enc.text_projection

        # forward
        if not self.use_side:
            return vis_outs, txt, state
        output = vis_outs, txt, state, outputs

        return output

    def output_shapes(self):
        return {
            i: ShapeSpec(channels=768)
            for i in range(self.num_layers + 1)
        }