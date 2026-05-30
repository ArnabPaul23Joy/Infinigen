# H2O ("Heavy-Hitter Oracle") variant of the vanilla baseline in selfattention.py.
#
# Differences vs selfattention.py at a glance:
#   - Constructor takes an extra `h2o_ratio` controlling how many heavy-hitter
#     tokens are retained.
#   - KV cache is grown dynamically (not pre-allocated to a fixed 2048 slots).
#   - Maintains a running per-token attention accumulator `self.acc` and an
#     iteration counter `self.i` (used to detect prefill).
#   - After prefill, prunes the KV cache to only the top-k tokens by
#     accumulated attention score (`_heavy_hitter_pruning`, new method).
#   - During decode, evicts the *least* important cached token (argmin of
#     `self.acc`) and writes the new K/V in its place, so the cache size
#     stays bounded regardless of how many tokens are generated.
import torch
from torch import nn
from typing import Tuple

class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        h2o_ratio: float,  # NEW vs selfattention.py: fraction of tokens kept as heavy-hitters
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))

        # H2O-only state. selfattention.py has no equivalent of these:
        #   acc — running sum of attention scores per cached token; used to
        #         rank tokens by cumulative importance for eviction.
        #   ratio — kept fraction (e.g. 0.2 keeps top 20% as heavy-hitters).
        #   i — step counter; i == 0 marks the prefill call.
        # past_key_value starts as None and is built as a (key, value) tuple
        # during the first forward, unlike selfattention.py which pre-allocates
        # a single (2, bsz, num_heads, 2048, head_dim) tensor up front.
        self.acc = None
        self.ratio = h2o_ratio
        self.i = 0
        self.past_key_value = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # NEW vs selfattention.py: the entire heavy-hitter selection step.
    # Called once right after prefill. Picks the `hh_k` cached tokens with the
    # highest aggregate attention weight (summed across query positions) per
    # head, drops the rest, and returns the pruned K/V plus the truncated
    # accumulator. This is the core of the H2O algorithm — capping cache size
    # to `hh_k` heavy-hitter tokens so memory does not grow with sequence
    # length.
    def _heavy_hitter_pruning(self, k, v, attn_weights, hh_k):
        # k, v: (s, b * n_head, head_dim)
        # attn_weights: (b * n_head, s, s)
        aggr_attn = torch.sum(attn_weights, 1)
        # (b * n_head, hh_k)
        _, topk_indices = aggr_attn[:, :].topk(
            min(hh_k, aggr_attn.shape[1]), dim=1)

        # select heavy-hitters
        # k, v: (b * n_head, s, head_dim)
        k_t = k.transpose(1, 0)
        v_t = v.transpose(1, 0)
        dim0_indices = torch.arange(k_t.size(0))[:, None]
        dim0_indices = dim0_indices.expand_as(topk_indices)
        # (b * n_head, hh_k, head_dim)
        k_hh_t = k_t[dim0_indices, topk_indices]
        v_hh_t = v_t[dim0_indices, topk_indices]
        # (hh_k, b * n_head, head_dim)
        k = k_hh_t.transpose(1, 0)
        v = v_hh_t.transpose(1, 0)
        # new shape (hh_k, b * n_head)
        aggr_attn = aggr_attn.transpose(0, 1)
        dim1_indices = torch.arange(aggr_attn.size(1)).unsqueeze(0)
        # (hh_k * 2, b * n_head)
        acc = aggr_attn[topk_indices.transpose(0, 1), dim1_indices]
        return k, v, acc

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()
        # NOTE vs selfattention.py: that file tracks `self.src_s` to know
        # where to write the new token into a pre-allocated cache. H2O does
        # not need it because the cache is fixed-size (hh_k + 1) and the
        # write slot is determined by the eviction rule below.

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # get key/value proj
        if self.past_key_value is not None:
            # DECODE path. selfattention.py writes the new K/V into the next
            # sequential slot; H2O always writes into the *last* slot (index
            # -1), which is the placeholder zero-row appended after pruning.
            # The H2O block at the bottom of forward() then decides which
            # cached row this new entry should permanently replace.
            k = self._shape(self.k_proj(hidden_states), -1, bsz).squeeze()
            v = self._shape(self.v_proj(hidden_states), -1, bsz).squeeze()
            key_states = self.past_key_value[0]
            key_states[:, :, -1] = k
            value_states = self.past_key_value[1]
            value_states[:, :, -1] = v
        else:
            # PREFILL path. selfattention.py allocates a (2, bsz, num_heads,
            # 2048, head_dim) buffer and copies the prompt K/V into it here.
            # H2O does NOT allocate the cache yet — it waits until after
            # softmax so it can use the prefill attention scores to choose
            # heavy-hitters before storing anything.
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # update kv cache
        #past_key_value = (key_states, value_states)

        # reshape
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # qkt
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # masking
        # DIFF vs selfattention.py: that file detects prefill via
        #   `attn_weights.shape[1] > 1` (multi-token query). H2O uses its own
        # step counter `self.i` because after prefill the cache is shrunk to
        # `hh_k` tokens — the shape-based check would still see multi-row
        # attn_weights occasionally and misfire.
        if self.i == 0: # prefill
            mask = torch.triu(torch.ones(attn_weights.shape).to('cuda'), diagonal=1) * -10000
            attn_weights = attn_weights + mask

        # softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)

        # sv
        attn_output = torch.bmm(attn_weights, value_states)

        # reshape
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        ##### h2o ####
        # ENTIRE BLOCK IS NEW vs selfattention.py.
        # selfattention.py simply returns here — it keeps every cached token.
        # H2O instead manages the cache: (a) prune to heavy-hitters right
        # after prefill, (b) evict-and-replace on each decode step.
        if self.acc is None:
            # First call (right after prefill). Compute how many heavy-hitter
            # tokens to retain and call _heavy_hitter_pruning to shrink the
            # cache to that size. A single zero-padded slot is appended so
            # the next decode step has somewhere to write the new K/V.
            self.hh = int(attn_weights.shape[-1] * self.ratio)
            key_states, value_states, self.acc = self._heavy_hitter_pruning(key_states.permute(1,0,2), value_states.permute(1,0,2), attn_weights, self.hh)
            key_states = key_states.permute(1, 0, 2)
            value_states = value_states.permute(1, 0, 2)
            self.past_key_value = (torch.cat((key_states.reshape(bsz, self.num_heads, key_states.shape[-2], key_states.shape[-1]), torch.zeros(bsz, self.num_heads, 1, key_states.shape[-1]).to('cuda').to(torch.float16)), dim = -2),
                              torch.cat((value_states.reshape(bsz, self.num_heads, value_states.shape[-2], value_states.shape[-1]), torch.zeros(bsz, self.num_heads, 1, key_states.shape[-1]).to('cuda').to(torch.float16)), dim = -2))

        else:
            # Subsequent decode calls. selfattention.py would append the new
            # K/V at the next free slot, growing the cache unboundedly. H2O
            # instead keeps the cache size fixed:
            #   1. Add this step's attention scores into `acc` (running
            #      importance score per cached token).
            #   2. Find the token with the smallest cumulative score —
            #      that is the least-important entry and the eviction victim.
            #   3. Overwrite the victim's K/V row and accumulator entry with
            #      the values from the last (newly added) row, then drop the
            #      now-redundant last row.
            temp_attn = attn_weights.squeeze(1).transpose(0, 1)
            self.acc = torch.cat((self.acc, torch.zeros(1, bsz * self.num_heads).to('cuda')), dim=0)
            self.acc = self.acc + temp_attn
            kick_ind = self.acc.argmin(dim=0).squeeze()

            # reduce accumulated result
            indices = kick_ind.unsqueeze(0)
            self.acc.scatter_(0, indices, self.acc[-1].unsqueeze(0).clone())
            self.acc = self.acc[:-1]

            # modify kv cache
            indices = kick_ind.view(-1, 1).expand(-1, self.head_dim).unsqueeze(1)
            key_states.scatter_(1, indices, key_states[:, -1].unsqueeze(1))
            value_states.scatter_(1, indices, value_states[:, -1].unsqueeze(1))
            #key_states = key_states[:, :-1]
            #value_states = value_states[:, :-1]
            self.past_key_value = (key_states.reshape(bsz, self.num_heads, key_states.shape[-2], key_states.shape[-1]),
                              value_states.reshape(bsz, self.num_heads, value_states.shape[-2], value_states.shape[-1]))

        # NEW vs selfattention.py: step counter advance, used by the prefill
        # detection at the top of the next forward() call.
        self.i += 1
        return attn_output
