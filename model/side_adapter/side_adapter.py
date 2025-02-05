import logging
from functools import partial
from typing import Dict, List, Tuple

import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.utils.logger import log_first_n
from detectron2.utils.registry import Registry
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.nn import functional as F

from ..layers import MLP, build_fusion_layer
from .timm_wrapper import PatchEmbed
import model.absvit
import math

SIDE_ADAPTER_REGISTRY = Registry("SIDE_ADAPTER")
SIDE_ADAPTER_REGISTRY.__doc__ = """
Registry for side adapter.
"""


def build_side_adapter_network(cfg, input_shape):
    name = cfg.SIDE_ADAPTER.NAME
    print(name)
    return SIDE_ADAPTER_REGISTRY.get(name)(cfg, input_shape)


class MLPMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        total_heads: int = 1,
        total_layers: int = 1,
        embed_channels: int = 256,
        mlp_channels: int = 512,
        mlp_num_layers: int = 3,
        rescale_attn_bias: bool = False,
    ):
        super().__init__()
        self.total_heads = total_heads
        self.total_layers = total_layers

        dense_affine_func = partial(nn.Conv2d, kernel_size=1)
        # Query Branch
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels, mlp_num_layers)
        self.multihead_attn1 = nn.MultiheadAttention(embed_channels,
                                                    4,
                                                    dropout=0.1,
                                                    kdim=embed_channels,
                                                    vdim=embed_channels)
        self.multihead_attn2 = nn.MultiheadAttention(embed_channels,
                                                    4,
                                                    dropout=0.1,
                                                    kdim=embed_channels,
                                                    vdim=embed_channels)
        # Pixel Branch
        self.pix_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )
        # Attention Bias Branch
        # self.attn_mlp = MLP(
        #     in_channels,
        #     mlp_channels,
        #     embed_channels * self.total_heads * self.total_layers,
        #     mlp_num_layers,
        #     affine_func=dense_affine_func,
        # )
        # if rescale_attn_bias:
        #     self.bias_scaling = nn.Linear(1, 1)
        # else:
        #     self.bias_scaling = nn.Identity()

    def forward(
        self, word, vis_pos, txt_pos, query: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # query: [B,N,C]
        # x: [B,C,H,W]
        query = self.query_mlp(query)
        pix = self.pix_mlp(x)
        b, c, h, w = pix.shape
        pix = pix.reshape(b, c, h*w)
        query = query.permute(1,0,2)
        pix = pix.permute(2,0,1)
        word = word.permute(1,0,2)
    

        pix = self.multihead_attn1(query=pix+vis_pos.to(pix.device),
                                   key=query,
                                   value=query)[0]
        mask_preds = self.multihead_attn2(query=pix+vis_pos.to(query.device),
                                   key=word+txt_pos.to(word.device),
                                   value=word)[0]
        # # preidict mask
        # mask_preds = torch.einsum("bqc,bchw->bqhw", query, pix)
        # # generate attn bias
        # attn = self.attn_mlp(x)
        # attn = attn.reshape(b, self.total_layers, self.total_heads, c, h, w)
        # attn_bias = torch.einsum("bqc,blnchw->blnqhw", query, attn)
        # attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)
        # attn_bias = attn_bias.chunk(self.total_layers, dim=1)
        # attn_bias = [attn.squeeze(1) for attn in attn_bias]
        return mask_preds.permute(1,2,0)#, attn_bias


@SIDE_ADAPTER_REGISTRY.register()
class RegionwiseSideAdapterNetwork(nn.Module):
    @configurable
    def __init__(
        self,
        vit_model: VisionTransformer,
        fusion_layers: nn.ModuleList,
        num_queries: int,
        fusion_map: Dict[int, int],
        deep_supervision_idxs: List[int],
    ):
        super().__init__()
        # remove cls token
        if vit_model.cls_token is not None:
            vit_model.pos_embed = nn.Parameter(vit_model.pos_embed[:, 1:, ...])
        del vit_model.cls_token
        vit_model.cls_token = None
        # delete out norm
        del vit_model.norm
        vit_model.norm = nn.Identity()
        self.vit_model = vit_model

        self.num_queries = num_queries
        self.num_features = vit_model.num_features
        # add query token
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, self.num_features))
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, num_queries, self.num_features)
        )
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        self.fusion_layers = fusion_layers
        self.fusion_map = fusion_map
        # self.mask_decoder = mask_decoder
        # for training
        self.deep_supervision_idxs = deep_supervision_idxs
        self.proj  = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_features)
        )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        vit = create_model(
            cfg.SIDE_ADAPTER.VIT_NAME,
            cfg.SIDE_ADAPTER.PRETRAINED,
            img_size=cfg.SIDE_ADAPTER.IMAGE_SIZE,
            num_classes=0,
            drop_rate=0,
            drop_path_rate=cfg.SIDE_ADAPTER.DROP_PATH_RATE,
        )
        # ["0->0","3->1","6->2","9->3"]
        fusion_map: List[str] = cfg.SIDE_ADAPTER.FUSION_MAP

        x2side_map = {int(j): int(i) for i, j in [x.split("->") for x in fusion_map]}
        # build fusion layers
        fusion_type: str = cfg.SIDE_ADAPTER.FUSION_TYPE
        fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": build_fusion_layer(
                    fusion_type, input_shape[src_idx].channels, vit.num_features
                )
                for tgt_idx, src_idx in x2side_map.items()
            }
        )
        # build mask decoder
        return {
            "vit_model": vit,
            "num_queries": cfg.SIDE_ADAPTER.NUM_QUERIES,
            "fusion_layers": fusion_layers,
            "fusion_map": x2side_map,
            "deep_supervision_idxs": cfg.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS,
        }

    def forward(
        self, image: torch.Tensor, clip_features: List[torch.Tensor], word
    ) -> Dict[str, List[torch.Tensor]]:
        features = self.forward_features(image, clip_features, word)
        return features
        # return self.decode_masks(features, word, vis_pos, txt_pos)

    def decode_masks(
        self, features: List[Dict[str, torch.Tensor]], word, vis_pos, txt_pos
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        if not self.training:
            features = [features[-1]]
        mask_preds = []
        # attn_biases = []
        for feature in features:
            # mask_pred, attn_bias = self.mask_decoder(**feature)
            mask_pred = self.mask_decoder(word, vis_pos, txt_pos, **feature)
            mask_preds.append(mask_pred)
            # attn_biases.append(attn_bias)
        return mask_preds#, attn_biases

    def forward_features(
        self, image: torch.Tensor, clip_features: List[torch.Tensor], word
    ) -> List[Dict[str, torch.Tensor]]:
        x = self.vit_model.patch_embed(image)
        td_flag = False
        if isinstance(x,tuple):
            td_flag = True
            x, (h,w) = x
        else:
            h = w = int(math.sqrt(x.shape[1]))
        L = x.shape[1]  # token length
        pos_embed = self.vit_model.pos_embed

        ori_h, ori_w = self.vit_model.patch_embed.grid_size
        if pos_embed.shape[1] != L:
            pos_embed = (
                F.interpolate(
                    pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                    size=[h, w],
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .permute(0, 2, 1)
            )
        pos_embed = torch.cat(
            [self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1
        )
        x = torch.cat(
            [self.query_embed.expand(x.shape[0], -1, -1), x],
            dim=1,
        )  # B, Q+L, C
        x = x + pos_embed

        prompt = self.proj(word)

        x = torch.cat(
            [x[:,:-L,:], prompt, x[:,-L:,:]],
            dim=1,
        )
        if td_flag:
            x = self.vit_model.norm_pre(x)
        x = self.fuse(0, x, clip_features, (h, w))
        outs = []

        # for top down attention
        input = x
        # first feedforward
        for i, blk in enumerate(self.vit_model.blocks):
            x = blk(x)
        x = self.vit_model.norm(x)
        # global prior
        prompt = prompt.mean(dim=1)
        cos_sim = F.normalize(x, dim=-1) @ F.normalize(prompt[None, ..., None], dim=1)  # B, N, 1c
        mask = cos_sim.clamp(0, 1)
        x = x * mask

        top_down_transform = prompt[..., None] @ prompt[..., None].transpose(-1, -2)
        x = x @ top_down_transform * 5

        td = self.vit_model.feedback(x)
        # final feedforward
        x = input
        in_var = []
        out_var = []
        for i, blk in enumerate(self.vit_model.blocks, start=1):
            # for top down attention
            in_var.append(x)
            x = blk(x, td[i-1])
            out_var.append(x)
            # x = blk(x)
            x = self.fuse(i, x, clip_features, (h, w))
            if i in self.deep_supervision_idxs:
                outs.append(
                    {
                        "query": x[:, :self.num_queries, ...],
                        "x": x[:, -L:, ...]
                        .permute(0, 2, 1)
                        .reshape(x.shape[0], x.shape[-1], h, w),
                    }
                )
            # if i < len(self.vit_model.blocks):
            #     x = x + pos_embed
        output = self.vit_model.norm(x)
        var_loss = self.var_loss(in_var, out_var, output[:, 0], self.proj(word.detach()))
        return outs, var_loss

    def fuse(
        self,
        block_idx: int,
        x: torch.Tensor,
        clip_features: List[torch.Tensor],
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if block_idx in self.fusion_map:
            src_idx = self.fusion_map[block_idx]
            L = spatial_shape[0] * spatial_shape[1]
            x = torch.cat(
                [
                    x[:, :-L, ...], # query 
                    self.fusion_layers[f"layer_{block_idx}"](
                        x[:, -L:, ...], clip_features[src_idx], spatial_shape
                    ),
                ],
                dim=1,
            )
            log_first_n(
                logging.INFO,
                f"fuse clip {src_idx} to {block_idx}",
                len(self.fusion_map),
            )
        return x
    def var_loss(self, in_var, out_var, img_emb, text_emb):
        recon_loss = []
        for depth in range(len(self.vit_model.decoders) - 1, -1, -1):
            recon, out = self.vit_model.decoders[depth](out_var[depth].detach())
            target = in_var[depth].detach()
            recon_loss.append(F.mse_loss(recon, target))
        return sum(recon_loss)