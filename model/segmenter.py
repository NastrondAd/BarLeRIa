import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import Neck, Decoder, Projector
from .intertwiner import Intertwiner_ViT, Intertwiner_EVA
from .side_adapter import build_side_adapter_network
from snf import set_snf
import eva_clip


class BARLERIA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        if "EVA" not in cfg.clip_pretrain:
            clip_model = torch.jit.load(cfg.clip_pretrain,
                                        map_location="cpu").eval()
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size).float()
            self.intertwiner = Intertwiner_ViT(d_model=cfg.ladder_dim, nhead=cfg.nhead, use_side=cfg.use_side, use_txt_add=cfg.use_txt_add)
        else:
            self.backbone,_,_ = eva_clip.create_model_and_transforms(cfg.model_name, pretrained=cfg.clip_pretrain, force_custom_clip=True, rope_size=cfg.input_size)
            self.intertwiner = Intertwiner_EVA(d_model=cfg.ladder_dim, nhead=cfg.nhead, use_side=cfg.use_side, use_txt_add=cfg.use_txt_add)
        
        if cfg.use_snf:
            set_snf(self.backbone.visual, cfg.snf_method, flow_length=cfg.flow_length)
            if 'EVA' not in cfg.clip_pretrain:
                set_snf(self.backbone.transformer, cfg.snf_method, flow_dim=cfg.word_dim, flow_length=cfg.flow_length)
            else:
                set_snf(self.backbone.text, cfg.snf_method, flow_dim=cfg.word_dim, flow_length=cfg.flow_length)
        # Fix Backbone
        for param_name, param in self.backbone.named_parameters():
            if 'positional_embedding' not in param_name and 'flow' not in param_name:
                param.requires_grad = False       

        # Multi-Modal Decoder
        self.neck = Neck(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = Decoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)

        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

        # Global Side Tuning
        self.use_side = cfg.use_side
        if self.use_side:
             self.side_linear = nn.Linear(144*len(cfg.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS), cfg.word_dim)
             self.side_adapter_network = build_side_adapter_network(cfg, self.intertwiner.output_shapes)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        if self.use_side:
            vis, word, state, outputs = self.intertwiner(img, word, self.backbone)
        else:
            vis, word, state = self.intertwiner(img, word, self.backbone)
        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()

        if self.use_side:
            fq_side=[]
            mask_preds, var_loss = self.side_adapter_network(
                img, outputs, word
            )
            for mask_pred in mask_preds:
                x = mask_pred['x']
                fq_side.append(x)
            fq_side = self.side_linear(torch.cat(fq_side,dim=1).permute(0,2,3,1)).permute(0,3,1,2)
            fq = self.decoder(fq+fq_side, word, pad_mask)
        else:
            fq = self.decoder(fq, word, pad_mask)
            var_loss = 0
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask) + var_loss
            return pred.detach(), mask, loss
        else:
            return pred.detach()
