from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

from .embeddings import apply_rotary_emb


def slice_rope_emb(emb, seqlen: int):
    """Trim cached cosine and sine tensors to shape (seqlen, head_dim)."""
    cos, sin = emb
    # slice sequence on dim=0
    cos = cos[:seqlen, :]
    sin = sin[:seqlen, :]
    return (cos, sin)


class CombinedGeoEmbedding(nn.Module):
    def __init__(self, geo_in_dim: int = 12, freq_dim: int = 256, embedding_dim: int = 3072):
        super().__init__()
        # Encode each relative camera matrix entry with freq_dim sinusoidal features.
        self.geo_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)

        # Project the flattened geometry encoding to the model width.
        self.timestep_embedder = TimestepEmbedding(
            in_channels=geo_in_dim * freq_dim,
            time_embed_dim=embedding_dim
        )

    def forward(self, rel_pose: torch.Tensor) -> torch.Tensor:
        # rel_pose: [B, S, 12]
        B, S, D = rel_pose.shape

        # Encode matrix entries independently: [B, S, 12, freq_dim].
        geo_freq = self.geo_proj(rel_pose.reshape(-1)).view(B, S, D, -1)

        # Flatten the matrix and frequency dimensions before the MLP.
        geo_freq = geo_freq.flatten(2)  # [B, S, 12 * freq_dim]
        return self.timestep_embedder(geo_freq.to(rel_pose.dtype))  # [B, S, embedding_dim]


class ViewMoEAttnProcessor:
    def __call__(
        self,
        attn: "ViewMoEAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        # Projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape [B, N, H, D]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Normalize queries and keys per attention head.
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Apply RoPE
        if image_rotary_emb is not None:
            # Separate rotary embeddings for generated queries and reference keys.
            q_emb, k_emb = image_rotary_emb

            q_emb = slice_rope_emb(q_emb, query.shape[1])
            k_emb = slice_rope_emb(k_emb, key.shape[1])

            query = apply_rotary_emb(query, q_emb, sequence_dim=q_emb[0].dim())
            key = apply_rotary_emb(key, k_emb, sequence_dim=k_emb[0].dim())

        # Attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        # Output Projection
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class ViewMoEAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(key_dim, inner_dim, bias=True)

        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

        # Start as an identity residual so fine-tuning initially preserves the base model.
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.processor = ViewMoEAttnProcessor()

    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb=None):
        return self.processor(self, hidden_states, encoder_hidden_states, image_rotary_emb=image_rotary_emb)


class ViewMoE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        inner_dim: int = 3072,         # Generated-token and geometry embedding width.
        cond_dim: int = 2048,          # Reference feature channels.
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        geo_freq_dim: int = 64,  # Sinusoidal features per camera matrix entry.
        top_k: int = 4,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.temperature = temperature
        self.k = top_k

        # Encode source cameras relative to the target for view routing.
        self.geo_embedder = CombinedGeoEmbedding(
            geo_in_dim=12,
            freq_dim=geo_freq_dim,
            embedding_dim=inner_dim
        )

        # Score each reference view from its relative camera geometry.
        self.gating = nn.Sequential(
            nn.Linear(inner_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )

        # Inject selected reference features into generated tokens.
        self.attn = ViewMoEAttention(
            query_dim=inner_dim,
            key_dim=cond_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim
        )

    def _to_homogeneous(self, m):
        last_row = torch.tensor([[0, 0, 0, 1]], device=m.device, dtype=m.dtype)
        last_row = last_row.expand(*m.shape[:-2], 1, 4)
        return torch.cat([m, last_row], dim=-2)

    def _get_relative_matrix_flat(self, source_w2c, target_w2c):
        B, S = source_w2c.shape[:2]
        source_h = self._to_homogeneous(source_w2c)
        target_h = self._to_homogeneous(target_w2c)
        # Invert in float32 for stability with scene-normalized camera matrices.
        target_h_inv = torch.inverse(target_h.squeeze(1).float()).to(target_h.dtype)
        rel_matrix = source_h @ target_h_inv.unsqueeze(1)
        return rel_matrix[:, :, :3, :].reshape(B, S, 12)

    def forward(
        self,
        noisy_state: torch.Tensor,
        recon_feats: torch.Tensor,
        target_w2c: torch.Tensor,
        source_w2c: torch.Tensor,
        image_rotary_emb: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_state: [B, N1, inner_dim]
            recon_feats: [B, S, N2, cond_dim]
            target_w2c: [B, 1, 3, 4]
            source_w2c: [B, S, 3, 4]
            image_rotary_emb: ((q_cos, q_sin), (k_cos, k_sin)).

        Returns:
            Refined generated tokens [B, N1, inner_dim] and routing logits [B, S].
        """
        B, S, N2, C = recon_feats.shape

        # Route references by their geometry relative to the target.
        rel_pose = self._get_relative_matrix_flat(source_w2c, target_w2c)  # [B,S,12]
        gate_input = self.geo_embedder(rel_pose)                           # [B,S,inner_dim]

        logits = self.gating(gate_input).squeeze(-1)               # [B,S]
        logits = logits / self.temperature

        actual_k = min(self.k, S)
        topk_v, topk_i = torch.topk(logits, k=actual_k, dim=-1)    # [B,k], [B,k]

        # softmax only over top-k
        topk_w = F.softmax(topk_v, dim=-1)                         # [B,k]

        # gather feats
        idx = topk_i[:, :, None, None].expand(B, actual_k, N2, C)
        topk_feats = recon_feats.gather(dim=1, index=idx)          # [B,k,N2,C]

        # gate K/V tokens
        topk_feats = topk_feats * topk_w[:, :, None, None].to(topk_feats.dtype)

        # flatten view dimension into token dimension for cross-attn
        encoder_hidden_states = topk_feats.reshape(B, actual_k * N2, C)     # [B,k*N2,C]

        # Refine generated tokens by attending to the selected references.
        refined_noisy = self.attn(
            hidden_states=noisy_state,                 # [B,N1,inner_dim]
            encoder_hidden_states=encoder_hidden_states,  # [B,k*N2,C]
            image_rotary_emb=image_rotary_emb
        )

        hidden_states = noisy_state + refined_noisy

        return hidden_states, logits
