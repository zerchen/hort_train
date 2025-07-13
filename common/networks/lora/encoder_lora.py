import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.all_head_size = qkv.all_head_size

    def forward(self, x, head_mask=None, output_attentions=False) -> torch.Tensor:
        # Compute the original qkv
        mixed_query_layer = self.qkv.query(x)
        key_layer = self.qkv.key(x)
        value_layer = self.qkv.value(x)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # Add new q and v components to the original qkv tensor
        mixed_query_layer += new_q
        value_layer += new_v

        key_layer = self.qkv.transpose_for_scores(key_layer)
        value_layer = self.qkv.transpose_for_scores(value_layer)
        query_layer = self.qkv.transpose_for_scores(mixed_query_layer)

        context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=head_mask, dropout_p=(self.qkv.dropout.p if self.training else 0.0), scale=1/math.sqrt(self.qkv.attention_head_size))

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = ((context_layer, attention_probs) if output_attentions else (context_layer,))

        return outputs


class DINOV2EncoderLoRA(nn.Module):
    def __init__(self, encoder, r: int = 64):
        super().__init__()
        assert r > 0

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Add LoRA layers to the encoder
        self.lora_layers = list(range(len(self.encoder.model.encoder.layer)))
        self.w_a = []
        self.w_b = []

        for i, block in enumerate(self.encoder.model.encoder.layer):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attention.attention
            dim = w_qkv_linear.all_head_size

            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attention.attention = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        feature = self.encoder(x, cond)

        return feature
