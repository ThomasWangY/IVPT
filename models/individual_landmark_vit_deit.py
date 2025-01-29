# Compostion of the VisionTransformer class from timm with extra features: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Union, Sequence
from timm.models import create_model
from timm.models.vision_transformer import Block, Attention
from utils.misc_utils import compute_attention

from .layers.transformer_layers import BlockWQKVReturn, AttentionWQKVReturn
from .layers.independent_mlp import IndependentMLPs
import torch.nn.functional as F


class IndividualLandmarkViT(torch.nn.Module):
    def __init__(self, init_model: torch.nn.Module, num_classes: int = 200,
                 part_dropout: float = 0.3, return_transformer_qkv: bool = False,
                 modulation_type: str = "original", gumbel_softmax: bool = False,
                 gumbel_softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 classifier_type: str = "linear", noise_variance: float = 0.0, n_pro: str = "") -> None:
        super().__init__()
        self.n_pro = [int(n) for n in n_pro.split(',')]
        self.num_landmarks = self.n_pro[-1] - 1 # num_landmarks
        self.num_classes = num_classes
        self.noise_variance = noise_variance
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.dist_token
        self.layer_n = len(self.n_pro) - 1
        self.gumbel_softmax = gumbel_softmax

        self.feature_dim = init_model.embed_dim
        self.patch_embed = init_model.patch_embed
        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm
        self.return_transformer_qkv = return_transformer_qkv
        self.h_fmap = int(self.patch_embed.img_size[0] // self.patch_embed.patch_size[0])
        self.w_fmap = int(self.patch_embed.img_size[1] // self.patch_embed.patch_size[1])

        self.p_token = nn.ParameterList([nn.Parameter(torch.zeros(1, self.n_pro[i], init_model.cls_token.shape[-1])) for i in range(self.layer_n+1)])
        for i, p in enumerate(self.p_token):
            nn.init.normal_(p, std=0.05)
        self.p_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.n_pro[i], self.h_fmap, self.w_fmap)) for i in range(self.layer_n+1)])
        self.p_linear = nn.ModuleList([nn.Linear(in_features=init_model.cls_token.shape[-1], out_features=init_model.cls_token.shape[-1], bias=True) for i in range(self.layer_n)])
        self.p_norm = nn.ModuleList([torch.nn.LayerNorm([self.num_landmarks, self.feature_dim]) for i in range(self.layer_n)])
        self.p_classifier = nn.ModuleList([nn.Linear(in_features=init_model.cls_token.shape[-1], out_features=self.num_landmarks, bias=True) for i in range(self.layer_n)]) # TODO

        self.unflatten = nn.Unflatten(1, (self.h_fmap, self.w_fmap))
        self.gumbel_softmax_temperature = gumbel_softmax_temperature
        self.gumbel_softmax_hard = gumbel_softmax_hard
        self.modulation_type = modulation_type
        self.modulation = torch.nn.LayerNorm([self.feature_dim, self.num_landmarks + 1])
        self.dropout_full_landmarks = torch.nn.Dropout1d(part_dropout)
        self.classifier_type = classifier_type
        if classifier_type == "independent_mlp":
            self.fc_class_landmarks = IndependentMLPs(part_dim=self.num_landmarks, latent_dim=self.feature_dim,
                                                      num_lin_layers=1, act_layer=False, out_dim=num_classes,
                                                      bias=False, stack_dim=1)
        elif classifier_type == "linear":
            self.fc_class_landmarks = torch.nn.Linear(in_features=self.feature_dim, out_features=num_classes,
                                                      bias=False)
        else:
            raise ValueError("classifier_type not implemented")
        self.convert_blocks_and_attention()
        self._init_weights()

    def _init_weights_head(self):
        # Initialize weights with a truncated normal distribution
        if self.classifier_type == "independent_mlp":
            self.fc_class_landmarks.reset_weights()
        else:
            torch.nn.init.trunc_normal_(self.fc_class_landmarks.weight, std=0.02)
            if self.fc_class_landmarks.bias is not None:
                torch.nn.init.zeros_(self.fc_class_landmarks.bias)

    def _init_weights_linear(self):
        for layer in self.p_linear:
            torch.nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def _init_weights(self):
        self._init_weights_head()
        self._init_weights_linear()

    def convert_blocks_and_attention(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.__class__ = BlockWQKVReturn
            elif isinstance(module, Attention):
                module.__class__ = AttentionWQKVReturn

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def compute_xq(self, x, q):
        ab = torch.einsum('bchw,blc->blhw', x, q)
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, q.shape[1], -1, -1).contiguous()
        a_sq = q.pow(2).sum(-1, keepdim=True)
        a_sq = a_sq.expand(-1, -1, x.shape[-2] * x.shape[-1])
        a_sq = a_sq.view(x.shape[0], q.shape[1], x.shape[-2], x.shape[-1])
        dist = b_sq - 2 * ab + a_sq
        maps = -dist
        return maps
    
    def compute_feat(self, maps, x):
        N = maps.shape[1]
        one_hot_map = F.one_hot(torch.argmax(maps, dim=1), num_classes=N).permute(0, 3, 1, 2)*maps
        all_features = (one_hot_map.unsqueeze(1) * x.unsqueeze(2)).contiguous()
        sum_pool = all_features.sum(dim=(3, 4)).permute(0, 2, 1)
        count_map = one_hot_map.sum(dim=(2, 3), keepdim=True)
        count_map_ = count_map + (count_map == 0).float()
        all_features = sum_pool / count_map_.squeeze(-1)
        return all_features, count_map

    def forward(self, x: Tensor) -> tuple[Any, Any, Any, Any, int | Any] | tuple[Any, Any, Any, Any, int | Any]:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x_len = x.shape[1]
        x_buffer, q_buffer, m_buffer, qm_buffer = [], [], [], []
        l = len(self.blocks)
        for i, block in enumerate(self.blocks):
            if i < l - self.layer_n:
                x = block(x)
            else:
                x = x[:, :x_len]
                q_index = i - l + self.layer_n
                q = self.p_token[q_index].expand(x.shape[0], -1, -1) # + q_pre.detach()
                x_buffer.append(x)
                q_buffer.append(q)
                # q = q.detach() # TODO
                x_ = self.norm(x.detach()) 
                x_ = x_[:, self.num_prefix_tokens:, :]  # [B, L, D]
                x_ = self.unflatten(x_)  # [B, H, W, D]
                x_ = x_.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

                maps = self.compute_xq(x_, q) # [B, P+1, H, W]
                p_bias = self.p_bias[q_index].expand(x.shape[0], -1, -1, -1)
                maps = torch.nn.functional.softmax(maps+p_bias, dim=1) # 

                q_x, count_map = self.compute_feat(maps.detach(), x_) # [B, P+1, D] [B, P+1, 1, 1] # 
                q_c = self.p_classifier[q_index](q_x[:, :-1]) # [B, P, N]

                count_mask = (count_map[:, :-1]==0).squeeze(-1).expand(-1, -1, q_c.shape[-1]) # [B, P, N]

                if self.gumbel_softmax:
                    q_maps = torch.nn.functional.gumbel_softmax(q_c, dim=-1, tau=self.gumbel_softmax_temperature, hard=self.gumbel_softmax_hard)
                else:
                    q_maps = torch.nn.functional.softmax(q_c, dim=-1)
                q_maps = q_maps.masked_fill(count_mask, 1/q_c.shape[-1])
                qm_buffer.append(q_maps)
                
                q_maps_expanded = q_maps.unsqueeze(-1)  # (B, P, N, 1)
                f_maps = maps.flatten(2) # (B, P+1, L)

                q_maps_detach = q_maps_expanded.detach()
                q_x_weighted = q_x[:, :-1].unsqueeze(2) * q_maps_detach  # (B, P, 1, D) × (B, P, N, 1) = (B, P, N, D)
                q_x = q_x_weighted.sum(dim=1)  # (B, N, D)
                q_maps_sum = q_maps_detach.sum(1).squeeze(1)
                q_x = q_x / q_maps_sum

                f_bg = f_maps[:, -1:] # (B, 1, L)
                f_maps_weighted = f_maps[:, :-1].unsqueeze(2) * q_maps_expanded  # (B, P, 1, L) × (B, P, N, 1) = (B, P, N, L)
                f_maps = f_maps_weighted.sum(dim=1)  # (B, N, L)

                f_maps = torch.cat([f_maps, f_bg], dim=1) # (B, N+1, L)
                
                m_buffer.append(f_maps)
                q_x = self.p_linear[q_index](self.p_norm[q_index](q_x))
                x = torch.cat([x, q_x], dim=1)

                x = block(x)

        q = self.p_token[-1].expand(x.shape[0], -1, -1)
        x = x[:, :x_len]
        x_buffer.append(x)
        q_buffer.append(q)

        maps_list = []
        for i, (x, q) in enumerate(zip(x_buffer, q_buffer)):
            x_ = self.norm(x.detach())
            x_ = x_[:, self.num_prefix_tokens:, :]  # [B, num_patch_tokens, embed_dim]
            x_ = self.unflatten(x_)  # [B, H, W, embed_dim]
            x_ = x_.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, H, W]
            
            maps = self.compute_xq(x_, q)
            
            p_bias = self.p_bias[i].expand(x.shape[0], -1, -1, -1)
            maps = torch.nn.functional.softmax(maps+p_bias, dim=1)  # [B, num_landmarks + 1, H, W] # 
            maps = maps + 1e-6
            maps = maps / maps.sum(dim=1, keepdim=True)
            maps_list.append(maps)

        maps = maps_list[-1]
        f_maps = maps.flatten(2)
        m_buffer.append(f_maps)

        x = self.unflatten(self.norm(x_buffer[-1])[:, self.num_prefix_tokens:, :]).permute(0, 3, 1, 2).contiguous()
        all_features = self.compute_feat(maps, x)[0]
        all_features = all_features.permute(0, 2, 1)

        # Modulate the features
        all_features_mod = self.modulation(all_features)  # [B, embed_dim, num_landmarks + 1]

        # Classification based on the landmark features
        scores = self.fc_class_landmarks(
            all_features_mod[..., :-1].permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        map_feat = maps_list

        return all_features_mod, maps_list[-3], scores, map_feat, (m_buffer, qm_buffer)

    def get_specific_intermediate_layer(
            self,
            x: torch.Tensor,
            n: int = 1,
            return_qkv: bool = False,
            return_att_weights: bool = False,
    ):
        num_blocks = len(self.blocks)
        attn_weights = []
        if n >= num_blocks:
            raise ValueError(f"n must be less than {num_blocks}")

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        if n == -1:
            if return_qkv:
                raise ValueError("take_indice cannot be -1 if return_transformer_qkv is True")
            else:
                return x

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)

                if return_att_weights:
                    attn_weight, _ = compute_attention(qkv)
                    attn_weights.append(attn_weight.detach())
            else:
                x = blk(x)
            if i == n:
                output = x.clone()
                if self.return_transformer_qkv and return_qkv:
                    qkv_output = qkv.clone()
                break
        if self.return_transformer_qkv and return_qkv and return_att_weights:
            return output, qkv_output, attn_weights
        elif self.return_transformer_qkv and return_qkv:
            return output, qkv_output
        elif self.return_transformer_qkv and return_att_weights:
            return output, attn_weights
        else:
            return output

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        if self.return_transformer_qkv:
            qkv_outputs = []
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)
            else:
                x = blk(x)
            if i in take_indices:
                outputs.append(x)
                if self.return_transformer_qkv:
                    qkv_outputs.append(qkv)
        if self.return_transformer_qkv:
            return outputs, qkv_outputs
        else:
            return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> tuple[tuple, Any]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        if self.return_transformer_qkv:
            outputs, qkv = self._intermediate_layers(x, n)
        else:
            outputs = self._intermediate_layers(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return_out = tuple(zip(outputs, prefix_tokens))
        else:
            return_out = tuple(outputs)

        if self.return_transformer_qkv:
            return return_out, qkv
        else:
            return return_out


def ivpt_vit_bb(backbone, img_size=224, num_cls=200, k=8, **kwargs):
    base_model = create_model(
        backbone,
        pretrained=False,
        img_size=img_size,
    )

    model = IndividualLandmarkViT(base_model, num_landmarks=k, num_classes=num_cls,
                                  modulation_type="layer_norm", gumbel_softmax=True,
                                  modulation_orth=True)
    return model


def ivptnet_vit_bb(backbone, img_size=224, num_cls=200, k=8, **kwargs):
    base_model = create_model(
        backbone,
        pretrained=False,
        img_size=img_size,
    )

    model = IndividualLandmarkViT(base_model, num_landmarks=k, num_classes=num_cls,
                                  modulation_type="original")
    return model
