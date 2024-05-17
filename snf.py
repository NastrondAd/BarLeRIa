import torch
from torch import nn
import timm
import math
from torch.nn import functional as F
from einops import rearrange
from flow import NormalizingFlow
import model.clip as clip
import eva_clip

def forward_flow_block(self, x):
    res_x, log_jacobians = self.flow(x)
    x = res_x + self.attention(self.ln_1(x))
    
    res_x, log_jacobians = self.flow2(x)
    x = res_x + self.mlp(self.ln_2(x))
   
    return x
def forward_text_block_eva(self, x, attn_mask=None):
    x = self.flow(x)[0] + self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask))
    x = self.flow2(x)[0] + self.ls_2(self.mlp(self.ln_2(x)))
    return x
    
def forward_block_eva(self, x, rel_pos_bias=None, attn_mask=None):
    x = x.permute(1, 0, 2)  # LND -> NLD
    if self.gamma_1 is None:
        if self.postnorm:
            x = self.flow(x)[0] + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))) 
            x = self.flow2(x)[0] + self.drop_path(self.norm2(self.mlp(x))) 
        else:
            x = self.flow(x)[0] + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) 
            x = self.flow2(x)[0] + self.drop_path(self.mlp(self.norm2(x)))
    else:
        if self.postnorm:
            x = self.flow(x)[0] + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))) 
            x = self.flow2(x)[0] + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x))) 
        else:
            x = self.flow(x)[0] + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) 
            x = self.flow2(x)[0] + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) 
    return x.permute(1, 0, 2)  # NLD -> LND


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def set_snf(model, method, flow_dim=768, xavier_init=False, flow_length=16):
    if method == 'normalization_flow':
        count = 0
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block or type(_) == clip.ResidualAttentionBlock:
                count+=1
                if count <= 12:
                    _.flow = NormalizingFlow(dim=flow_dim, flow_length=flow_length)
                    _.flow2 = NormalizingFlow(dim=flow_dim, flow_length=flow_length)
                    bound_method = forward_flow_block.__get__(_, _.__class__)
                    setattr(_, 'forward', bound_method)
            elif type(_) == eva_clip.eva_vit_model.Block:
                _.flow = NormalizingFlow(dim=flow_dim, flow_length=flow_length)
                _.flow2 = NormalizingFlow(dim=flow_dim, flow_length=flow_length)
                bound_method = forward_block_eva.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)

            elif type(_) == eva_clip.transformer.ResidualAttentionBlock:
                _.flow = NormalizingFlow(dim=flow_dim, flow_length=flow_length)
                _.flow2 = NormalizingFlow(dim=flow_dim, flow_length=flow_length)
                bound_method = forward_text_block_eva.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)

            elif len(list(_.children())) != 0:
                set_snf(_, method, flow_dim, xavier_init, flow_length)  