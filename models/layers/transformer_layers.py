# Attention Block with option to return the mean of k over heads from attention

import torch
from timm.models.vision_transformer import Attention, Block
import torch.nn.functional as F
from typing import Tuple


class AttentionWQKVReturn(Attention):
    """
    Modifications:
         - Return the qkv tensors from the attention
    """

    def forward(self, x_q, x_k=None, x_v=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_q
        B, NQ, C = x_q.shape
        B, NK, C = x_k.shape
        qkv = self.qkv(x_q).reshape(B, NQ, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]

        qkv = self.qkv(x_k).reshape(B, NK, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = qkv[1]

        qkv = self.qkv(x_v).reshape(B, NK, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = qkv[2]

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, NQ, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # torch.stack((q, k, v), dim=0)
        return x, (q, k, v)

class BlockWQKVReturn(Block):
    """
    Modifications:
        - Use AttentionWQKVReturn instead of Attention
        - Return the qkv tensors from the attention
    """

    def forward(self, x_q, x_k=None, x_v=None, return_qkv: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_q
        x_attn, qkv = self.attn(self.norm1(x_q), self.norm1(x_k), self.norm1(x_v))
        x = x_q + self.drop_path1(self.ls1(x_attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if return_qkv:
            return x, qkv
        else:
            return x
