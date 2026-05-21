import random
import os

try:
    from moba_attn_zhao import moba_attn_varlen
except ImportError:
    moba_attn_varlen = None
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from flash_attn.modules.mha import MHA

# from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

# from nn_common import check_nan, check_nan_is_all
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# import fairscale.nn.model_parallel.initialize as fs_init
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )



world_size = 1
rank = 0

# @dataclass
# class ModelArgs:
#     dim: int = 4096
#     n_layers: int = 2
#     n_heads: int = 2
#     n_kv_heads: Optional[int] = None
#     vocab_size: int = -1
#     multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
#     ffn_dim_multiplier: Optional[float] = None
#     norm_eps: float = 1e-5
#     rope_theta: float = 500000
#
#     max_batch_size: int = 32
#     max_seq_len: int = 2048

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.attention(query, key, value, attn_mask=mask)
        return attn_output

class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, cross_attn = False, use_flash_attn = True,return_residual = False):
        super().__init__()
        self.cross_attn = cross_attn
        self.return_residual = return_residual
        self.attention = MHA(embed_dim, num_heads, cross_attn = cross_attn, use_flash_attn = use_flash_attn, return_residual=return_residual)

    def forward(self, x, x_kv=None, mask=None):
        if not self.cross_attn:
            attn_output = self.attention(x, x_kv=None, key_padding_mask=mask)
        else:
            assert x_kv is not None
            attn_output = self.attention(x, x_kv=x_kv,  key_padding_mask=mask)
        return attn_output if not self.return_residual else (attn_output, x)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, inter_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, embed_dim)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.dropout(x) # remove dropout
        x = self.fc2(x)
        return x

class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Router, self).__init__()
        self.gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Compute expert probabilities
        expert_logits = self.gating(x)  # Shape: [batch_size, num_experts]
        expert_probs = F.softmax(expert_logits, dim=-1)
        return expert_probs

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, capacity):
        """
        Mixture of Experts layer.
        :param input_dim: Dimensionality of input features.
        :param output_dim: Dimensionality of output features.
        :param num_experts: Number of experts.
        :param capacity: Number of top experts to use for each input.
        """
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.router = Router(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass for Mixture of Experts.
        :param x: Input tensor of shape (batch_size, seq_length, input_dim) or (batch_size, input_dim).
        :return: Output tensor of shape (batch_size, seq_length, output_dim) or (batch_size, output_dim).
        """
        if x.dim() == 2:  # Case: (batch_size, input_dim)
            batch_size, input_dim = x.size()
            seq_length = 1
            x = x.unsqueeze(1)  # Add a dummy sequence length dimension
        elif x.dim() == 3:  # Case: (batch_size, seq_length, input_dim)
            batch_size, seq_length, input_dim = x.size()
        else:
            raise ValueError(f"Unsupported input dimensions: {x.shape}")

        # Flatten sequence and batch dimensions for routing
        x_flat = x.view(batch_size * seq_length, input_dim)

        # Compute expert probabilities
        expert_probs = self.router(x_flat)  # Shape: [batch_size * seq_length, num_experts]
        #debug
        # print("MoE.forward: x_flat shape:", x_flat.shape)
        # print("MoE.forward: expert_probs stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        #     expert_probs.min().item(), expert_probs.max().item(), expert_probs.mean().item()))
        
        # Get top-k experts and their probabilities
        top_k = torch.topk(expert_probs, self.capacity, dim=-1)
        selected_experts = top_k.indices  # Shape: [batch_size * seq_length, capacity]
        selected_probs = top_k.values  # Shape: [batch_size * seq_length, capacity]
        # #debug
        # print("MoE.forward: selected_experts shape:", selected_experts.shape)
        # print("MoE.forward: selected_probs stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        #     selected_probs.min().item(), selected_probs.max().item(), selected_probs.mean().item()))

        # Prepare outputs
        output_dim = self.experts[0].out_features
        outputs = torch.zeros(batch_size * seq_length, output_dim, device=x.device)

        # Process top-k experts
        for i in range(self.capacity):
            expert_index = selected_experts[:, i]  # Indices of the selected expert for each sample
            expert_weight = selected_probs[:, i]  # Probabilities of the selected expert
            
            print(f"MoE.forward: Expert {j} processes {sample_indices.numel()} samples")
            
            # Route inputs to selected experts
            expert_outputs = torch.cat([
                self.experts[j](x_flat[expert_index == j])  # Apply expert `j` to the relevant inputs
                if (expert_index == j).sum() > 0 else torch.zeros(0, output_dim, device=x.device)
                for j in range(self.num_experts)
            ], dim=0)

            # Add weighted outputs
            outputs += expert_outputs * expert_weight.unsqueeze(-1)

        # Reshape outputs back to original dimensions
        outputs = outputs.view(batch_size, seq_length, output_dim)
        if seq_length == 1:  # Remove sequence length dimension if it was added
            outputs = outputs.squeeze(1)

        return outputs

class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, embed_dim, n_routed_experts, n_activated_experts, n_expert_groups, n_limited_groups, score_func="softmax", route_scale=1.0):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = embed_dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, embed_dim))

        # Initialize weights with Xavier/Glorot initialization
        torch.nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.empty(n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        # check_nan(x, f'x in gate')

        scores = F.linear(x, self.weight)

        # check_nan(x, f'scores in gate before softmax')

        if self.score_func == "softmax":
            scores = scores - scores.max(dim=-1, keepdim=True)[0]  # ADDED, Subtract max for numerical stability
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        # check_nan(scores, f'scores in gate')

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        # check_nan(scores, f'weights before clamp in gate')

        weights = torch.clamp(weights, min=1e-7)  # ADDED, Avoid very small/zero values
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        # check_nan(scores, f'weights after clamp in gate')

        return weights.type_as(x), indices

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, expert_dropout=0.0):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = embed_dim
        # assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // world_size
        self.n_activated_experts = n_activated_experts
        self.expert_dropout = expert_dropout
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(embed_dim, n_routed_experts, n_activated_experts, n_expert_groups=1, n_limited_groups=1, score_func="softmax", route_scale=1.0)
        self.experts = nn.ModuleList([Expert(embed_dim, moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = FeedForward(embed_dim, n_shared_experts * moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        修改后的 MoE.forward：
          1. 将输入展平为 (N, dim)。
          2. 通过 gate 得到每个样本的 topk 专家索引和对应权重（形状均为 [N, topk]）。
          3. 对于每个专家（遍历所有 n_routed_experts），用非零 mask 找出该专家被选择的所有样本，
             并用 index_add_ 聚合专家输出的加权结果。
          4. 加上 shared专家输出后恢复原始形状。
        """
        shape = x.size()
        x_flat = x.view(-1, self.dim)  # [N, dim]
        weights, indices = self.gate(x_flat)  # weights, indices: [N, topk]
        N, topk = weights.shape

        if self.training and self.expert_dropout > 0:
            keep = torch.rand_like(weights) > self.expert_dropout
            keep[:, 0] = True
            weights = weights * keep
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        y = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_routed_experts):
            # 找出哪些样本选择了 expert_idx
            mask = (indices == expert_idx)  # [N, topk] boolean
            if mask.sum() == 0:
                continue
            # nonzero 返回形状 [M, 2]，第一列为样本索引，第二列为该样本在 topk 中的位置
            sel = mask.nonzero(as_tuple=False)
            sample_indices = sel[:, 0]  # [M]
            # 对应使用的权重（每个样本可能出现多次则自动累加）
            weight_values = weights[mask].unsqueeze(-1)  # [M, 1]
            # 计算 expert 输出
            expert = self.experts[expert_idx]
            # 注意：如果某些样本可能出现多次，则对相同样本的 expert 输出进行累加
            expert_output = expert(x_flat[sample_indices])  # [M, dim]
            # 累加结果到 y：对于相同 sample index, 将加权输出相加
            y.index_add_(0, sample_indices, expert_output * weight_values)
        # 计算共享专家输出
        z = self.shared_experts(x_flat)
        out_flat = y + z
        return out_flat.view(shape)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, capacity):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.cross_attn = Attention(embed_dim, num_heads)
        # self.ff = FeedForward(embed_dim, ff_dim)
        self.moe = MixtureOfExperts(embed_dim, embed_dim, num_experts, capacity)
        # self.moe = MOE()
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # Cross-attention
        cross_attn_output = self.cross_attn(x, cross_input, cross_input, mask)
        x = self.norm2(x + cross_attn_output)

        # Feedforward and MoE
        # ff_output = self.ff(x)
        moe_output = self.moe(x)
        # x = self.norm3(x + ff_output + moe_output)
        x = self.norm3(x + moe_output)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=8192):
        """
        Sinusoidal Positional Encoding module.
        :param embed_dim: The dimension of the embeddings.
        :param max_seq_len: The maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # Create position encoding matrix
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        encoding = torch.zeros(max_seq_len, embed_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        encoding = encoding.unsqueeze(0)  # Add batch dimension: (1, max_seq_len, embed_dim)
        self.register_buffer('positional_encoding', encoding)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.
        :param x: Input tensor of shape (batch_size, seq_len, embed_dim).
        :return: Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        # return x + self.positional_encoding[:, :seq_len]
        return self.positional_encoding[:, :seq_len]

# implement a transformer by using the transformer block above
class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, num_experts, capacity, seq_length):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(seq_length, embed_dim)
        print(self.embedding)
        self.pos_embedding = PositionalEncoding( embed_dim, max_seq_len=seq_length)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, num_experts, capacity) for _ in range(num_layers)
        ])

    def forward(self,x, cross_input, mask=None):
        x = self.embedding(x) + self.pos_embedding(x)

        # x = self.embedding(x)  # Convert indices to embeddings
        for layer in self.layers:
            x = layer(x,cross_input, mask)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, inter_dim = 10944, layer_id=None):
        super(Block, self).__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.n_dense_layers = 3
        n_dense_layers = self.n_dense_layers

        if layer_id is not None:
            self.moe = FeedForward(embed_dim, inter_dim) if layer_id < n_dense_layers else MoE(embed_dim,
                                                                                               n_routed_experts,
                                                                                               n_activated_experts,
                                                                                               n_shared_experts,
                                                                                               moe_inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts,
                           n_shared_experts,
                           moe_inter_dim)

        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input=None, mask=None):
        x_norm = self.norm1(x)
        if cross_input is None:
            # Self-attention
            attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
            attn_output = x + attn_output
        else:
            # Cross-attention
            cross_input_norm = self.norm1(cross_input)
            attn_output = self.self_attn(x_norm, cross_input_norm, cross_input_norm, mask)
            attn_output = x + attn_output

        # check_nan(attn_output, 'attn_output in self_attn')

        moe_output = self.moe(self.norm3(attn_output))
        moe_output = attn_output + moe_output

        return moe_output

class BlockMoba(nn.Module):
    """
    替换原Block的标准注意力为moba注意力，保留MoE逻辑和同样的init参数
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        inter_dim=10944,
        layer_id=None,
        moba_chunk_size=5,
        moba_topk=2
    ):
        super(BlockMoba, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk

        self.moe = MoE(
            embed_dim,
            n_routed_experts,
            n_activated_experts,
            n_shared_experts,
            moe_inter_dim,
            expert_dropout=0.2,
        )

        # 与原Block保持一致的归一化
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input=None, mask=None):
        """
        x: [batch, seq_len, embed_dim]
        cross_input: 如果需要 cross-attention，可以在内部做相应区分
        mask: [batch, seq_len, seq_len], 可选
        """
        x_norm = self.norm1(x)

        if cross_input is None:
            # 自注意力
            attn_output = self._moba_attention(x_norm, x_norm, x_norm, mask)
        else:
            # cross-attention（自行决定是否也用moba）
            cross_input_norm = self.norm1(cross_input)
            attn_output = self._moba_attention(x_norm, cross_input_norm, cross_input_norm, mask)

        # 残差
        out = x + attn_output

        # MoE / FF 处理
        moe_output = self.moe(self.norm3(out))
        out = out + moe_output

        return out

    # def _moba_attention(self, q, k, v, mask=None):
    #     """
    #     用moba_attn_varlen替换原先标准/flash注意力
    #     """
    #     bsz, seqlen, d_model = q.shape
    #     head_dim = d_model // self.num_heads
    #     assert d_model % self.num_heads == 0, "embed_dim与num_heads不匹配"

    #     # flatten后调用moba_attn_varlen
    #     q_reshaped = q.reshape(bsz * seqlen, self.num_heads, head_dim)
    #     k_reshaped = k.reshape(bsz * seqlen, self.num_heads, head_dim)
    #     v_reshaped = v.reshape(bsz * seqlen, self.num_heads, head_dim)

    #     cu_seqlens = torch.arange(0, bsz * seqlen + 1, seqlen, device=q.device, dtype=torch.int32)
    #     attn_out, _ = moba_attn_varlen(
    #         q=q_reshaped,
    #         k=k_reshaped,
    #         v=v_reshaped,
    #         cu_seqlens=cu_seqlens,
    #         max_seqlen=seqlen,
    #         moba_chunk_size=self.moba_chunk_size,
    #         moba_topk=self.moba_topk,
    #         mask=mask
    #     )
    #     return attn_out.reshape(bsz, seqlen, d_model)
    # 在 BlockMoba 中修改 _moba_attention 方法
    def _moba_attention(self, q, k, v, mask=None):
        """使用标准注意力替代 moba_attn_varlen"""
        bsz, seqlen, d_model = q.shape
        head_dim = d_model // self.num_heads
        
        # 标准多头注意力实现
        q = q.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return attn_output
class BlockFlash(nn.Module):
    def __init__(self, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, inter_dim = 10944, layer_id=None, cross_attn = False):
        super().__init__()
        self.cross_attn = cross_attn
        self.self_attn = FlashAttention(embed_dim, num_heads,cross_attn = cross_attn, use_flash_attn = True, return_residual = False)
        self.n_dense_layers = 3
        n_dense_layers = self.n_dense_layers

        if layer_id is not None:
            self.moe = FeedForward(embed_dim, inter_dim) if layer_id < n_dense_layers else MoE(embed_dim,
                                                                                               n_routed_experts,
                                                                                               n_activated_experts,
                                                                                               n_shared_experts,
                                                                                               moe_inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts,
                           n_shared_experts,
                           moe_inter_dim)

        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input=None, mask=None):
        x_norm = self.norm1(x)
        if not self.cross_attn:
            # Self-attention
            attn_output = self.self_attn(x_norm, x_kv=None, mask=mask)
            attn_output = x + attn_output
        else:
            assert cross_input is not None
            # Cross-attention
            cross_input_norm = self.norm1(cross_input)
            attn_output =  self.self_attn(x_norm, x_kv=cross_input_norm, mask=mask)
            attn_output = x + attn_output

        # check_nan(attn_output, 'attn_output in self_attn')

        moe_output = self.moe(self.norm3(attn_output))
        moe_output = attn_output + moe_output

        return moe_output

class BlockNoCross(nn.Module):
    def __init__(self, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, inter_dim = 10944, layer_id=None):
        super(BlockNoCross, self).__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.n_dense_layers = 3
        n_dense_layers = self.n_dense_layers
        if layer_id is not None:
            self.moe =  FeedForward(embed_dim, inter_dim) if layer_id < n_dense_layers else MoE(embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts,
                           n_shared_experts,
                           moe_inter_dim)

        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, mask=None):
        # print('testing remove attention --------------------------------------------------')
        # Self-attention
        attn_output = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        attn_output = x + attn_output

        # MoE
        moe_output = self.moe(self.norm3(attn_output))
        moe_output = attn_output + moe_output

        return moe_output

class BlockCross(nn.Module):
    def __init__(self, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, inter_dim = 10944, layer_id=None):
        super().__init__()
        self.cross_attn = Attention(embed_dim, num_heads)
        self.n_dense_layers = 3
        n_dense_layers = self.n_dense_layers
        if layer_id is not None:
            self.moe =  FeedForward(embed_dim, inter_dim) if layer_id < n_dense_layers else MoE(embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts,
                           n_shared_experts,
                           moe_inter_dim)

        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input, mask=None):
        # print('testing remove attention --------------------------------------------------')
        # Self-attention
        cross_input_norm = self.norm1(cross_input)
        x_norm = self.norm1(x)
        attn_output = self.cross_attn(x_norm, cross_input_norm, cross_input_norm, mask)
        attn_output = x + attn_output

        # MoE
        moe_output = self.moe(self.norm3(attn_output))
        moe_output = attn_output + moe_output

        return moe_output

class BlockNoCrossFlash(nn.Module):
    def __init__(self, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, inter_dim = 10944, layer_id=None):
        super().__init__()
        self.self_attn = FlashAttention(embed_dim, num_heads)
        # self.self_attn = Attention(embed_dim, num_heads)
        # self.self_attn = MHA(embed_dim, num_heads, use_flash_attn=True)
        n_dense_layers = 3
        if layer_id is not None:
            self.moe =  FeedForward(embed_dim, inter_dim) if layer_id < n_dense_layers else MoE(embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts,
                           n_shared_experts,
                           moe_inter_dim)

        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, mask=None):
        # Self-attention
        # attn_output = self.self_attn(self.norm1(x))
        # print('testing removing attn------------------------------------')
        # attn_output2 = x

        attn_output = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        attn_output2 = torch.add(x.clone(), attn_output)  # Explicitly clone to avoid in-place operation


        # MoE
        moe_output = self.moe(self.norm3(attn_output2))
        moe_output2 = torch.add(attn_output2.clone(), moe_output)  # Explicitly clone to avoid in-place operation

        return moe_output2

class TransformerDeepSeek(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, seq_length, vocab_size =  100):
        super(TransformerDeepSeek, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # print(self.embedding)
        self.pos_embedding = PositionalEncoding(embed_dim, max_seq_len=seq_length)
        # print(self.pos_embedding)
        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim) for _ in range(num_layers)
        ])
        self.linear_head1 = nn.Linear(embed_dim, vocab_size)
        self.linear_head2 = nn.Linear(embed_dim, vocab_size)

    def forward(self,x1, cross_input, x2, mask=None):
        # x = self.embedding(x1) + self.pos_embedding(x1)
        # x = self.embedding(x1)  # Convert indices to embeddings
        x = self.embedding(x1) + self.embedding(x2)  # Convert indices to embeddings
        # print(x.shape)
        for layer in self.layers:
            x = layer(x, cross_input, mask)
        x1_output = self.linear_head1(x)
        x2_output = self.linear_head2(x)
        return x1_output, x2_output

class TransformerDeepSeek2(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, seq_length, vocab_size =  100):
        # global world_size, rank
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        # rank = dist.get_rank() if dist.is_initialized() else 0
        super(TransformerDeepSeek2, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # print(self.embedding)
        self.pos_embedding = PositionalEncoding(embed_dim, max_seq_len=seq_length)
        # print(self.pos_embedding)
        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim) for _ in range(num_layers)
        ])
        self.linear_head1 = nn.Linear(embed_dim, vocab_size)
        self.linear_head2 = nn.Linear(embed_dim, vocab_size)

        self.linear1 = nn.Linear(2*embed_dim, embed_dim)

    def forward(self,x1, x2, s1, s2, graph_embs, mask=None):
        """
        :param x1:
        :param x2:
        :param s1: s1 is the input for cross attention
        :param s2: s2 is the input for cross attention
        :param graph_embs:  graph_embs shall be the same shape as x1
        :param mask:
        :return:
        """
        # print('deep seek 2 forward')
        x = self.embedding(x1) + self.embedding(x2)  # Convert indices to embeddings
        # add positional encoding
        x = x + self.pos_embedding(x1) + self.pos_embedding(x2)
        # print('add graph ')
        x = torch.cat([x, graph_embs], dim=-1)

        x = self.linear1(x)
        # print('add cross attention')
        cross_input = self.embedding(s1) + self.embedding(s2)  # Convert indices to embeddings
        cross_input = cross_input + self.pos_embedding(s1) + self.pos_embedding(s2)  # Convert indices to embeddings

        check_nan(x, 'x')
        check_nan(cross_input, 'cross_input')

        # print('begin layers')
        for layer in self.layers:
            x = layer(x, cross_input, mask)
            check_nan(x, 'x in layer')
        x1_output = self.linear_head1(x)
        x2_output = self.linear_head2(x)

        # print('forward done')
        return x1_output, x2_output

class TransformerDeepSeek1(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, seq_length, vocab_size =  100):
        super(TransformerDeepSeek1, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # print(self.embedding)
        self.pos_embedding = PositionalEncoding(embed_dim, max_seq_len=seq_length)
        # print(self.pos_embedding)
        self.layers = nn.ModuleList([
            BlockNoCross(embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim) for _ in range(num_layers)
        ])
        self.linear_head1 = nn.Linear(embed_dim, vocab_size)
        self.linear_head2 = nn.Linear(embed_dim, vocab_size)

        self.linear1 = nn.Linear(2*embed_dim, embed_dim)
        # self.linear2 = nn.Linear(2*embed_dim, embed_dim)

    def forward(self,x1, x2, s1, s2, graph_embs, mask=None):
        """
        :param x1:
        :param x2:
        :param s1: s1 is the input for cross attention
        :param s2: s2 is the input for cross attention
        :param graph_embs:  graph_embs shall be the same shape as x1
        :param mask:
        :return:
        """
        # x = self.embedding(x1) + self.pos_embedding(x1)
        x = self.embedding(x1) + self.embedding(x2)  # Convert indices to embeddings
        # add positional encoding
        x = x + self.pos_embedding(x1) + self.pos_embedding(x2)
        # print('x', x.shape)
        # print('graph_embs', graph_embs.shape)
        x = torch.cat([x, graph_embs], dim=-1)
        x = self.linear1(x)

        s = self.embedding(s1) + self.embedding(s2)  # Convert indices to embeddings
        s = s + self.pos_embedding(s1) + self.pos_embedding(s2)

        x = torch.cat([x, s], dim=-2)
        # x = self.linear2(x)
        # print(x.shape)
        for layer in self.layers:
            x = layer(x, mask)
        x1_output = self.linear_head1(x)
        x2_output = self.linear_head2(x)
        return x1_output, x2_output

# only use sp data
class TransformerDeepSeek1Sp(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, seq_length, vocab_size =  100):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # print(self.embedding)
        self.pos_embedding = PositionalEncoding(embed_dim, max_seq_len=seq_length)
        # print(self.pos_embedding)
        self.layers = nn.ModuleList([
            BlockNoCross(embed_dim, num_heads, n_routed_efxperts, n_activated_experts, n_shared_experts, moe_inter_dim) for _ in range(num_layers)
        ])
        self.linear_head1 = nn.Linear(embed_dim, vocab_size)
        self.linear_head2 = nn.Linear(embed_dim, vocab_size)

        self.linear1 = nn.Linear(2*embed_dim, embed_dim)
        # self.linear2 = nn.Linear(2*embed_dim, embed_dim)

    def forward(self,x1, x2, graph_embs, mask=None):
        """
        :param x1:
        :param x2:
        :param graph_embs:  graph_embs shall be the same shape as x1
        :param mask:
        :return:
        """
        x = self.embedding(x1) + self.embedding(x2)  # Convert indices to embeddings
        # add positional encoding
        x = x + self.pos_embedding(x1) + self.pos_embedding(x2)
        # print('x', x.shape)
        # print('graph_embs', graph_embs.shape)
        x = torch.cat([x, graph_embs], dim=-1)
        x = self.linear1(x)

        for layer in self.layers:
            x = layer(x, mask)
        x1_output = self.linear_head1(x)
        x2_output = self.linear_head2(x)
        return x1_output, x2_output

# only use sp data
class TransformerDeepSeek1Sp2(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, seq_length, vocab_size =  100, inter_dim = 10944):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim, max_seq_len=seq_length)
        self.layers = nn.ModuleList([BlockNoCross(embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim,  inter_dim = inter_dim, layer_id=i) for i in range(num_layers)])
        # print('Testing with all linear')
        # self.layers = nn.ModuleList([BlockNoCross(embed_dim, num_heads, n_routed_experts, n_activated_experts,
        #                                           n_shared_experts, moe_inter_dim, inter_dim=inter_dim, layer_id=0) for
        #                              i in range(num_layers)
        #                              ])
        self.linear_head1 = nn.Linear(embed_dim, vocab_size)
        self.linear_head2 = nn.Linear(embed_dim, vocab_size)

        self.linear1 = nn.Linear(2*embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self,x1, x2, graph_embs, mask=None):
        """
        :param x1:
        :param x2:
        :param graph_embs:  graph_embs shall be the same shape as x1
        :param mask:
        :return:
        """

        x = self.embedding(x1) + self.embedding(x2)  # Convert indices to embeddings
        x = x + self.pos_embedding(x1) + self.pos_embedding(x2)

        x = torch.cat([x, graph_embs], dim=-1)
        x = self.linear1(x)

        for layer in self.layers:
            x = layer(x, mask)

        x1_output = self.linear_head1(x)
        x2_output = self.linear_head2(x)
        return x1_output, x2_output

class TransformerDeepSeek1Sp2Flash(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, seq_length, vocab_size =  100, inter_dim = 10944):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim, max_seq_len=seq_length)
        self.layers = nn.ModuleList([BlockNoCrossFlash(embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim,  inter_dim = inter_dim, layer_id=i) for i in range(num_layers)])
        # print('Testing with all linear')
        # self.layers = nn.ModuleList([BlockNoCrossFlash(embed_dim, num_heads, n_routed_experts, n_activated_experts,
        #                                           n_shared_experts, moe_inter_dim, inter_dim=inter_dim, layer_id=0) for
        #                              i in range(num_layers)
        #                              ])
        self.linear_head1 = nn.Linear(embed_dim, vocab_size)
        self.linear_head2 = nn.Linear(embed_dim, vocab_size)

        self.linear1 = nn.Linear(2*embed_dim, embed_dim)

    def forward(self,x1, x2, graph_embs, mask=None):
        """
        :param x1:
        :param x2:
        :param graph_embs:  graph_embs shall be the same shape as x1
        :param mask:
        :return:
        """

        x = self.embedding(x1) + self.embedding(x2)  # Convert indices to embeddings
        x = torch.add(x, self.pos_embedding(x1) + self.pos_embedding(x2))

        x = torch.cat([x, graph_embs], dim=-1)
        x = self.linear1(x)

        for layer in self.layers:
            x = layer(x, mask)

        x1_output = self.linear_head1(x)
        x2_output = self.linear_head2(x)
        return x1_output, x2_output


device = torch.device("cuda")

class TransformerCLS_Last_CellType(nn.Module):
    def __init__(self, model_conf, *, symbol_vocab_size=100, expr_max_bin=100, is_cross_first=False, is_cell_gene=True, cell_type_num = 2893):
        super().__init__()
        self.transformerCLS = TransformerCLS(model_conf,symbol_vocab_size=symbol_vocab_size, expr_max_bin=expr_max_bin, is_cross_first=is_cross_first, is_cell_gene=is_cell_gene)
        embed_dim = model_conf.embed_dim
        self.cell_type_output = nn.Linear(embed_dim, cell_type_num, bias=False)

    def forward(self, expr_val=None, gene_id=None, expr_val_pooling=None, gene_id_pooling=None, mask_x=None,
                n_neigh=None, data_str='sp', ESM2_emb=None, regulated_graph_emb=None):
        """
        :param expr_val: expression value
        :param gene_id: gene id indices
        :param expr_val_pooling: s1 is the input for cross attention, expression value indices
        :param gene_id_pooling: s2 is the input for cross attention, gene id indices
        :param graph_embs:  graph_embs shall be the same shape as x1
        :param mask_x:
        :return:
        """
        output_dic = self.transformerCLS(expr_val=expr_val, gene_id=gene_id, expr_val_pooling=expr_val_pooling, gene_id_pooling=gene_id_pooling, mask_x=mask_x, n_neigh = n_neigh, data_str = data_str, ESM2_emb = ESM2_emb, regulated_graph_emb = regulated_graph_emb)

        if data_str == 'sc':
            output_dic['cell_type_output'] = self.cell_type_output(output_dic['cell_embedding'])

        return output_dic
    
class TransformerDeepSeek2All_orig(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        inter_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        seq_length,
        symbol_vocab_size =  100, 
        expr_bin_vocab_size = 100,
        cross_all=True,
        use_flash_attn=False,
        gene_lengths_dim=1,  # 这个要删掉的不要管
        expr_max_bin=100,     # 新增
        wiki_embedding=None,   # 新增参数
        esm2_embedding=None  
    ):
        super().__init__()

        self.vocab_size = symbol_vocab_size
        self.seq_length = seq_length
        self.cross_all = cross_all
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.expr_max_bin = expr_max_bin
        self.symbol_embedding = nn.Embedding(symbol_vocab_size, embed_dim)
        self.special_token_embedding = nn.Embedding(2, embed_dim)
        self.non_zero_linear1 = nn.Linear(1, expr_max_bin)
        self.non_zero_silu = nn.SiLU()
        self.non_zero_linear2 = nn.Linear(expr_max_bin, expr_max_bin)
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bin_embedding = nn.Embedding(expr_max_bin, embed_dim)
        self.gene_id_emb = nn.Embedding(symbol_vocab_size, 1024, padding_idx=0)
        # self.pred_head1 = nn.Softplus()
        # self.pred_head1 = nn.ReLU()
        
        
        # 注册模态降维层，将所有模态统一降维到 512 维
        self.proj_gene_length = nn.Linear(1, 1024)  
        self.proj_wiki = nn.Linear(768, 1024)       
        self.proj_esm2 = nn.Linear(1280, 1024)       
        self.proj_graph1 = nn.Linear(512, 1024)  
        self.proj_graph2 = nn.Linear(512, 1024)  
        self.proj_x = nn.Linear(embed_dim, 1024) 
        # 在 TransformerDeepSeek2All.__init__ 中添加
        self.fusion_layer = nn.Linear(5120, embed_dim)
        # self.fusion_layer = nn.Linear(2*embed_dim, embed_dim)
        
        #注册权重系数
        # self.main_data_weight = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        # self.wiki_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # self.esm2_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # self.graph1_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # self.graph2_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        
        
        #注册layernorm层
        self.norm_gene = nn.LayerNorm(1024)
        self.norm_wiki = nn.LayerNorm(1024)
        self.norm_esm2 = nn.LayerNorm(1024)
        self.norm_graph1 = nn.LayerNorm(1024)
        self.norm_graph2 = nn.LayerNorm(1024)
        self.norm_x = nn.LayerNorm(1024)
        self.norm_gene_id = nn.LayerNorm(1024)
        self.norm_final__x= nn.LayerNorm(5120)

        # 组装 Transformer 层
        self.layers = nn.ModuleList([
            BlockMoba(
                embed_dim,
                num_heads,
                n_routed_experts,
                n_activated_experts,
                n_shared_experts,
                moe_inter_dim,
                inter_dim=inter_dim,
                layer_id=i,
                moba_chunk_size=5,
                moba_topk=2
            )
            for i in range(num_layers)
        ])

        # 预测头，任务为回归（输出 1 维），使用两个预测头
        self.linear_head1 = nn.Linear(2*embed_dim, 1)
        self.cell_type_classifier = nn.Linear(embed_dim,2984)
        self.tissue_classifier = nn.Linear(embed_dim,264)
        self.disease_classifier = nn.Linear(embed_dim, 108)
        # 添加输出激活函数和缩放因子
        self.output_activation = nn.ReLU()

        # 如果需要拼接图特征，则先映射到 embed_dim
        self.graph_proj = nn.Linear(embed_dim * 2, embed_dim)

        # 仅保留基因长度投影
        self.proj_gene = nn.Linear(gene_lengths_dim, embed_dim)

        # # 新增 total_proj 映射层：根据是否包含图特征使用不同输入维度
        # self.total_proj_with_graph = nn.Linear(3 * embed_dim, embed_dim)
        # self.total_proj_no_graph = nn.Linear(2 * embed_dim, embed_dim)

    def get_expr_embedding(self,expr_val,mask_val):
        # Identify zero vs. non-zero position
        zero_mask = (expr_val - 0.0) < 1e-3 # for float 16
        if mask_val is not None:
            mask_val_mask = (expr_val - mask_val) < 1e-3
        else:
            mask_val_mask = torch.zeros_like(expr_val, dtype=torch.bool)

        expr_val_nz_2d = expr_val.unsqueeze(-1)  # shape (Z,1)

        # set alpha dtype the same with expr_val dtype
        alpha_casted = self.alpha.to(expr_val.dtype)
        alpha_casted = alpha_casted.to(device=expr_val.device)
        v1 = self.non_zero_silu(self.non_zero_linear1(expr_val_nz_2d))
        v2 = self.non_zero_linear2(v1) + alpha_casted * expr_val_nz_2d
        v3 = F.softmax(v2, dim=-1)

        E = v3 @ self.bin_embedding.weight  # (Z, embed_dim)

        if zero_mask.any():
            E[zero_mask] = self.special_token_embedding.weight[0].to(E.dtype)

        if mask_val_mask.any():
            E[mask_val_mask] = self.special_token_embedding.weight[1].to(E.dtype)

        return E

    def forward(self, tr_gene_values, tr_gene_ids, expr_val_pooling, gene_id_pooling, *, 
            graph_embs=None, gene_length=None,
            wiki_embedding=None, esm2_embedding=None,   # 新增这两个参数
            new_graph_emb=None, new_graph_emb2=None, mask_x=None,
            new_graph_gene_ids=None, new_graph_gene_ids2=None, mask_val=-1.0):
        emb1 = self.get_expr_embedding(tr_gene_values, mask_val=mask_val)
        x = emb1
    
       # 4) 模态降维与融合
        if all(v is not None for v in [gene_length, new_graph_emb, new_graph_emb2]):
            gene_length = gene_length.unsqueeze(-1)  
            gene_feat = self.proj_gene_length(gene_length)
            wiki_feat = self.proj_wiki(wiki_embedding)
            esm2_feat = self.proj_esm2(esm2_embedding)
            # 利用扩展广播，不做重复复制：
            graph1_feat = self.proj_graph1(new_graph_emb)
            graph2_feat = self.proj_graph2(new_graph_emb2)
            x = self.proj_x(x)
       
            # gene_feat = self.norm_gene(gene_feat)
            # wiki_feat = self.norm_wiki(wiki_feat)
            # esm2_feat = self.norm_esm2(esm2_feat)
            # graph1_feat = self.norm_graph1(graph1_feat)
            # graph2_feat = self.norm_graph2(graph2_feat)
            # x = self.norm_x(x)
       
            if tr_gene_ids is not None:
                gene_id_emb = self.gene_id_emb(tr_gene_ids)
                gene_id_emb = self.norm_gene_id(gene_id_emb)
                # graph1_gene_id_emb = self.gene_id_emb(new_graph_gene_ids)
                # graph1_gene_id_emb = self.norm_gene_id(graph1_gene_id_emb)
                # graph2_gene_id_emb = self.gene_id_emb(new_graph_gene_ids2)
                # graph2_gene_id_emb = self.norm_gene_id(graph2_gene_id_emb)
       
                x = x + gene_id_emb

                # graph1_feat = graph1_feat + graph1_gene_id_emb
                # graph2_feat = graph2_feat + graph2_gene_id_emb
            # print(f"x dtype: {x.dtype}")
            # print(f"wiki_feat dtype: {wiki_feat.dtype}")
            # print(f"esm2_feat dtype: {esm2_feat.dtype}")
            # print(f"graph1_feat dtype: {graph1_feat.dtype}")
            # print(f"graph2_feat dtype: {graph2_feat.dtype}")

            # print("x shape before squeeze:", x.shape)
            # print("wiki_feat shape before squeeze:", wiki_feat.shape)
            # print("esm2_feat shape before squeeze:", esm2_feat.shape)
            # print("graph1_feat shape:", graph1_feat.shape)
            # print("graph2_feat shape:", graph2_feat.shape)
            x = torch.cat([
                x,
                wiki_feat,
                esm2_feat,
                graph1_feat,
                graph2_feat
            ], dim=-1)
            x = self.norm_final__x(x)
            x = self.fusion_layer(x)
        
        for layer in self.layers:
            x = layer(x, cross_input=None, mask=mask_x)

        if mask_val is not None:
            mask_val_mask = (tr_gene_values - mask_val) < 1e-3  # for float 16 (batch, seq)
        else:
            mask_val_mask = torch.zeros_like(tr_gene_values, dtype=torch.bool)
        mask_expanded = mask_val_mask.unsqueeze(-1)  # Shape: (batch, seq, 1)
        # x_unmasked = x * (~mask_expanded)
        # cell_embedding = x_unmasked.mean(dim=1,keepdim=True)

        # seq = tr_gene_values.shape[1]
        # ratio = seq/(~mask_expanded).sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1, 1)
        # ratio = ratio.to(x.dtype)
        # ratio = ratio.detach()
        # cell_embedding = cell_embedding * ratio
        
        # cell_embedding_single = cell_embedding[:, 0, :]  # (batch, embed_dim)
        # cell_type_logits = self.cell_type_classifier(cell_embedding_single)
       
        # cell_embedding = cell_embedding.expand(-1, seq, -1)  # (batch, seq, emb_dim)
        cell_embedding = x[:, 0, :]            # shape: (batch, embed_dim)
        cell_type_logits = self.cell_type_classifier(cell_embedding)
        tissue_logits = self.tissue_classifier(cell_embedding)
        disease_logits = self.disease_classifier(cell_embedding)
        # 扩展 CLS token 的维度：先 unsqueeze（增加序列维度），后 expand 到整个序列长度
        cell_embedding_expanded = cell_embedding.unsqueeze(1).expand(-1, x.size(1), -1)  # shape: (batch, seq, embed_dim)
       
        # 如果需要将 cell_embedding 拼接回序列中，则扩展后拼接
        # cell_embedding_expanded = cell_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        if graph_embs is not None:
            cell_embedding_expanded = cell_embedding_expanded + graph_embs
       
        x = torch.cat([x, cell_embedding_expanded], dim=-1)  # (batch, seq, emb_dim*2)

        # 6) 预测头
        x1_output = self.linear_head1(x)
        return x1_output,cell_type_logits,tissue_logits,disease_logits
    
    
class TransformerDeepSeek_gaze(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        inter_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        d_model=768,
        out_dim=3,
        dropout_rate=0.1,
        num_domains: int = 2,
        add_domain_to_all_tokens: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.add_domain_to_all_tokens = add_domain_to_all_tokens
        self.dropout = nn.Dropout(dropout_rate)  # 新增 Dropout
        self.proj_f1 = nn.Linear(512, d_model)
        self.proj_f2 = nn.Linear(512, d_model)
        self.proj_f3 = nn.Linear(2048, d_model)
        self.proj_patch = nn.Linear(d_model, d_model)
        # 域嵌入：简单的可学习向量（源域/目标域）
        self.domain_embed = nn.Embedding(num_domains, d_model)
        self.layers = nn.ModuleList([
            BlockMoba(
                d_model,
                num_heads,
                n_routed_experts,
                n_activated_experts,
                n_shared_experts,
                moe_inter_dim,
                inter_dim=inter_dim,
                layer_id=i,
                moba_chunk_size=5,
                moba_topk=2
            )
            for i in range(num_layers)
        ])
        self.linear_head = nn.Linear(d_model, out_dim)

    def forward(self, raw_inputs, domain_ids: torch.Tensor = None, mask=None, return_repr: bool = False):
        token_f1 = self.dropout(self.proj_f1(raw_inputs["feature_1"]).unsqueeze(1))
        token_f2 = self.dropout(self.proj_f2(raw_inputs["feature_2"]).unsqueeze(1))
        token_f3 = self.dropout(self.proj_f3(raw_inputs["feature_3"]))
        token_list = [token_f1, token_f2, token_f3]
        if "token_img_patch" in raw_inputs and raw_inputs["token_img_patch"] is not None:
            token_patch = self.dropout(self.proj_patch(raw_inputs["token_img_patch"]))
            token_list.append(token_patch)
        transformer_input = torch.cat(token_list, dim=1)
        # 注入域信息：简单加法
        if domain_ids is not None:
            domain_vec = self.domain_embed(domain_ids)  # [B, d_model]
            if self.add_domain_to_all_tokens:
                transformer_input = transformer_input + domain_vec.unsqueeze(1)
            else:
                transformer_input[:, 0, :] = transformer_input[:, 0, :] + domain_vec
        x = transformer_input
        for layer in self.layers:
            x = self.dropout(layer(x, cross_input=None, mask=mask))  # 在每层后加 Dropout
        global_repr = x.mean(dim=1)
        out = self.linear_head(global_repr)
        if return_repr:
            return out, global_repr
        return out
    

class TransformerDeepSeek2All(nn.Module):

    # 1. **参数与变量初始化**
    # - 模型构造函数接收多个超参数（层数、维度、头数、专家数量、序列长度、词表大小等）以及额外的模态参数（如 gene_length、wiki_embedding、esm2_embedding、图嵌入 new_graph_emb 等）。
    # - 定义诸多属性，包括主输入维度（embed_dim）、表达值离散化维度（expr_max_bin）、词表大小（symbol_vocab_size）等。

    # 2. **构建表达值嵌入分支**
    # - 初始化一个主嵌入：
    #     • 一个用于输入 token 转换的 nn.Embedding（symbol_embedding）。
    #     • 一个特殊 token 的嵌入（special_token_embedding），用于零表达或 mask 部分。
    # - 定义非零表达值处理分支：
    #     • 两个线性层（non_zero_linear1 与 non_zero_linear2）和激活函数（SiLU）处理输入表达值，将其映射到 expr_max_bin 的维度，然后通过 Softmax 得到概率分布；
    #     • 利用概率分布与预定义的 bin_embedding 进行加权求和，最终得到表达值的嵌入。

    # 3. **构建 gene id 嵌入与其他模态映射**
    # - 使用一个 gene id 嵌入层（gene_id_emb）产生 1024 维度表示，再用 proj_gene_id 将其映射到与主分支一致的 embed_dim。
    # - 同时，针对其他模态定义投影层：
    #     • proj_gene_length 将 gene_length（标量）映射到 embed_dim；
    #     • proj_wiki 将 Wiki 嵌入（768 维）映射到 embed_dim；
    #     • proj_esm2 将 ESM2 嵌入（1280 维）映射到 embed_dim；
    #     • proj_graph1 与 proj_graph2 将两组图嵌入映射到 embed_dim；
    #     • proj_x 将表达值主分支投影到 embed_dim。
    # - 为各个模态分别注册 LayerNorm 层（norm_gene、norm_wiki、norm_esm2、norm_graph1、norm_graph2、norm_x、norm_gene_id），确保特征数值稳定。

    # 4. **模态融合**
    # - 在 forward 中，如果 gene_length、图嵌入、wiki 与 esm2 数据均不为空，则先分别通过上一步的投影和归一化处理，得到对应的模态特征。
    # - 如果存在 gene id 信息，则将其嵌入加到主表达分支上。
    # - 将主分支、wiki、esm2、图1、图2 特征按通道拼接，再经过 fusion_layer（一个线性层）和归一化（norm_final__x）融合为统一维度的表示。

    # 5. **Transformer 层堆叠**
    # - 使用 nn.ModuleList 构建多个 BlockMoba 层，每一层内部已嵌入 moba 注意力和 MoE（混合专家）结构。
    # - 这些 Transformer 层对融合后的表示进行逐层传播和特征提炼。

    # 6. **后续输出与任务头**
    # - 从最后一层 Transformer 输出中取出 CLS token（通常取序列中第一个向量）作为 cell_embedding；
    # - 使用 cell_type_classifier 进行细胞类型分类；
    # - 将 cell_embedding 扩展到整个序列维度，并与 Transformer 输出拼接（即将主序列特征与全局 cell embedding 融合），再经过一个预测头 linear_head1 生成最终的回归或其他任务输出。

    # 7. **整体流水线回顾**
    # - 模型首先利用表达值、gene id 和其他辅助模态（gene_length、Wiki、ESM2、图）的输入，各自生成初步嵌入；
    # - 各模态经过各自投影与归一化后，在一处通过拼接和 fusion_layer 融合为统一表示；
    # - 融合后的表示经过多层 Transformer（BlockMoba 层）进行上下文特征提取；
    # - 最后，模型同时输出用于回归任务的预测结果（linear_head1 输出）和基于 CLS token 的细胞类型分类结果（cell_type_classifier 输出）。

    # 这种设计实现了多模态信息（表达值、基因 id、基因长度、Wiki、ESM2 与图嵌入）的联合建模，通过各个投影和归一化保证模态一致性，后续 Transformer 层对整体信息进行深度特征提取，最终输出任务所需的预测结果。
    def __init__(
            self,
            num_layers,
            embed_dim,
            inter_dim,
            num_heads,
            n_routed_experts,
            n_activated_experts,
            n_shared_experts,
            moe_inter_dim,
            symbol_vocab_size=100,
            expr_bin_vocab_size=100,
            expr_max_bin=100  # 新增
    ):
        super().__init__()

        self.vocab_size = symbol_vocab_size
        self.embed_dim = embed_dim
        self.expr_max_bin = expr_max_bin
        self.symbol_embedding = nn.Embedding(symbol_vocab_size, embed_dim)
        self.special_token_embedding = nn.Embedding(2, embed_dim)
        self.non_zero_linear1 = nn.Linear(1, expr_max_bin)
        self.non_zero_silu = nn.SiLU()
        self.non_zero_linear2 = nn.Linear(expr_max_bin, expr_max_bin)
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bin_embedding = nn.Embedding(expr_max_bin, embed_dim)
        self.gene_id_emb = nn.Embedding(symbol_vocab_size, 1024, padding_idx=0)
        # self.pred_head1 = nn.Softplus()
        # self.pred_head1 = nn.ReLU()

        # 注册模态降维层，将所有模态统一降维到 512 维
        self.proj_gene_length = nn.Linear(1, 1024)
        self.proj_wiki = nn.Linear(768, 1024)
        self.proj_esm2 = nn.Linear(1280, 1024)
        self.proj_graph1 = nn.Linear(512, 1024)
        self.proj_graph2 = nn.Linear(512, 1024)
        self.proj_x = nn.Linear(embed_dim, 1024)
        # 在 TransformerDeepSeek2All.__init__ 中添加
        # self.fusion_layer = nn.Linear(5120, embed_dim)
        # self.fusion_layer = nn.Linear(2*embed_dim, embed_dim)

        self.fusion_layer = nn.Linear(1024, embed_dim)


        # 注册权重系数
        # self.main_data_weight = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        # self.wiki_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # self.esm2_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # self.graph1_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # self.graph2_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # 注册layernorm层
        self.norm_gene = nn.LayerNorm(1024)
        self.norm_wiki = nn.LayerNorm(1024)
        self.norm_esm2 = nn.LayerNorm(1024)
        self.norm_graph1 = nn.LayerNorm(1024)
        self.norm_graph2 = nn.LayerNorm(1024)
        self.norm_x = nn.LayerNorm(1024)
        self.norm_gene_id = nn.LayerNorm(1024)
        self.norm_final__x = nn.LayerNorm(5120)

        # 组装 Transformer 层
        self.layers = nn.ModuleList([
            BlockMoba(
                embed_dim,
                num_heads,
                n_routed_experts,
                n_activated_experts,
                n_shared_experts,
                moe_inter_dim,
                inter_dim=inter_dim,
                layer_id=i,
                moba_chunk_size=5,
                moba_topk=2
            )
            for i in range(num_layers)
        ])

        # 预测头，任务为回归（输出 1 维），使用两个预测头
        self.linear_head1 = nn.Linear(2 * embed_dim, 1)
        self.cell_type_classifier = nn.Linear(embed_dim, 2984)
        self.tissue_classifier = nn.Linear(embed_dim, 264)
        self.disease_classifier = nn.Linear(embed_dim, 108)
        # 添加输出激活函数和缩放因子
        self.output_activation = nn.ReLU()

        # 如果需要拼接图特征，则先映射到 embed_dim
        self.graph_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.v_weights = nn.Parameter(torch.ones(6))


        # 仅保留基因长度投影
        # gene_lengths_dim = 1
        # self.proj_gene = nn.Linear(gene_lengths_dim, embed_dim)

        # # 新增 total_proj 映射层：根据是否包含图特征使用不同输入维度
        # self.total_proj_with_graph = nn.Linear(3 * embed_dim, embed_dim)
        # self.total_proj_no_graph = nn.Linear(2 * embed_dim, embed_dim)

    def get_expr_embedding(self, expr_val, mask_val):
        # Identify zero vs. non-zero position
        zero_mask = (expr_val - 0.0) < 1e-3  # for float 16
        if mask_val is not None:
            mask_val_mask = (expr_val - mask_val) < 1e-3
        else:
            mask_val_mask = torch.zeros_like(expr_val, dtype=torch.bool)

        expr_val_nz_2d = expr_val.unsqueeze(-1)  # shape (Z,1)

        # set alpha dtype the same with expr_val dtype
        alpha_casted = self.alpha.to(expr_val.dtype)
        alpha_casted = alpha_casted.to(device=expr_val.device)
        v1 = self.non_zero_silu(self.non_zero_linear1(expr_val_nz_2d))
        v2 = self.non_zero_linear2(v1) + alpha_casted * expr_val_nz_2d
        v3 = F.softmax(v2, dim=-1)

        E = v3 @ self.bin_embedding.weight  # (Z, embed_dim)

        if zero_mask.any():
            E[zero_mask] = self.special_token_embedding.weight[0].to(E.dtype)

        if mask_val_mask.any():
            E[mask_val_mask] = self.special_token_embedding.weight[1].to(E.dtype)

        return E

    def forward(self, tr_gene_values, tr_gene_ids, *,
                graph_embs=None, gene_length=None,
                wiki_embedding=None, esm2_embedding=None,  # 新增这两个参数
                new_graph_emb=None, new_graph_emb2=None, mask_x=None,
                new_graph_gene_ids=None, new_graph_gene_ids2=None, mask_val=-1.0):
        emb1 = self.get_expr_embedding(tr_gene_values, mask_val=mask_val)
        x = emb1

        # 4) 模态降维与融合
        if all(v is not None for v in [gene_length, new_graph_emb, new_graph_emb2]):
            # gene_length = gene_length.unsqueeze(-1)
            # gene_feat = self.proj_gene_length(gene_length)
            wiki_feat = self.proj_wiki(wiki_embedding)
            esm2_feat = self.proj_esm2(esm2_embedding)
            # 利用扩展广播，不做重复复制：
            graph1_feat = self.proj_graph1(new_graph_emb)
            graph2_feat = self.proj_graph2(new_graph_emb2)
            x = self.proj_x(x)

            # gene_feat = self.norm_gene(gene_feat)
            wiki_feat = self.norm_wiki(wiki_feat)
            esm2_feat = self.norm_esm2(esm2_feat)
            graph1_feat = self.norm_graph1(graph1_feat)
            graph2_feat = self.norm_graph2(graph2_feat)
            x = self.norm_x(x)

            gene_id_emb = self.gene_id_emb(tr_gene_ids)
            gene_id_emb = self.norm_gene_id(gene_id_emb)

            # if tr_gene_ids is not None:
            #     gene_id_emb = self.gene_id_emb(tr_gene_ids)
            #     gene_id_emb = self.norm_gene_id(gene_id_emb)
            #     # graph1_gene_id_emb = self.gene_id_emb(new_graph_gene_ids)
            #     # graph1_gene_id_emb = self.norm_gene_id(graph1_gene_id_emb)
            #     # graph2_gene_id_emb = self.gene_id_emb(new_graph_gene_ids2)
            #     # graph2_gene_id_emb = self.norm_gene_id(graph2_gene_id_emb)
            #
            #     x = x + gene_id_emb
            #
            #     # graph1_feat = graph1_feat + graph1_gene_id_emb
            #     # graph2_feat = graph2_feat + graph2_gene_id_emb
            # print(f"x dtype: {x.dtype}")
            # print(f"wiki_feat dtype: {wiki_feat.dtype}")
            # print(f"esm2_feat dtype: {esm2_feat.dtype}")
            # print(f"graph1_feat dtype: {graph1_feat.dtype}")
            # print(f"graph2_feat dtype: {graph2_feat.dtype}")

            # print("x shape before squeeze:", x.shape)
            # print("wiki_feat shape before squeeze:", wiki_feat.shape)
            # print("esm2_feat shape before squeeze:", esm2_feat.shape)
            # print("graph1_feat shape:", graph1_feat.shape)
            # print("graph2_feat shape:", graph2_feat.shape)
            # print('gene_id_emb shape:', gene_id_emb.shape)

            # x = torch.cat([
            #     x,
            #     wiki_feat,
            #     esm2_feat,
            #     graph1_feat,
            #     graph2_feat
            # ], dim=-1)
            # x = self.norm_final__x(x)
            # x = self.fusion_layer(x)

            norm_weights = torch.softmax(self.v_weights, dim=0)  # shape: [6]
            vectors = torch.stack([x, gene_id_emb, wiki_feat, esm2_feat, graph1_feat, graph2_feat],
                                  dim=0)  # shape: [6, dim]
            weighted_vectors = norm_weights[:, None, None, None] * vectors
            x = weighted_vectors.sum(dim=0)
            x = self.fusion_layer(x)

        for layer in self.layers:
            x = layer(x, cross_input=None, mask=mask_x)

        if mask_val is not None:
            mask_val_mask = (tr_gene_values - mask_val) < 1e-3  # for float 16 (batch, seq)
        else:
            mask_val_mask = torch.zeros_like(tr_gene_values, dtype=torch.bool)
        # mask_expanded = mask_val_mask.unsqueeze(-1)  # Shape: (batch, seq, 1)
        # x_unmasked = x * (~mask_expanded)
        # cell_embedding = x_unmasked.mean(dim=1,keepdim=True)

        # seq = tr_gene_values.shape[1]
        # ratio = seq/(~mask_expanded).sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1, 1)
        # ratio = ratio.to(x.dtype)
        # ratio = ratio.detach()
        # cell_embedding = cell_embedding * ratio

        # cell_embedding_single = cell_embedding[:, 0, :]  # (batch, embed_dim)
        # cell_type_logits = self.cell_type_classifier(cell_embedding_single)

        # cell_embedding = cell_embedding.expand(-1, seq, -1)  # (batch, seq, emb_dim)
        cell_embedding = x[:, 0, :]  # shape: (batch, embed_dim)
        cell_type_logits = self.cell_type_classifier(cell_embedding)
        tissue_logits = self.tissue_classifier(cell_embedding)
        disease_logits = self.disease_classifier(cell_embedding)
        # 扩展 CLS token 的维度：先 unsqueeze（增加序列维度），后 expand 到整个序列长度
        cell_embedding_expanded = cell_embedding.unsqueeze(1).expand(-1, x.size(1),
                                                                     -1)  # shape: (batch, seq, embed_dim)

        # 如果需要将 cell_embedding 拼接回序列中，则扩展后拼接
        # cell_embedding_expanded = cell_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        if graph_embs is not None:
            cell_embedding_expanded = cell_embedding_expanded + graph_embs

        x = torch.cat([x, cell_embedding_expanded], dim=-1)  # (batch, seq, emb_dim*2)

        # 6) 预测头
        x1_output = self.linear_head1(x)
        return x1_output, cell_type_logits, tissue_logits, disease_logits
    
class TransformerDeepSeekZhao2(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        inter_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        seq_length,
        symbol_vocab_size=100, 
        expr_bin_vocab_size=100,
        cross_all=True,
        use_flash_attn=False,
        expr_max_bin=100,     
        wiki_embedding=None,   
        esm2_embedding=None  
    ):
        super().__init__()
        self.vocab_size = symbol_vocab_size
        self.seq_length = seq_length
        self.cross_all = cross_all
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.expr_max_bin = expr_max_bin
        
        # 主输入部分（由表达值获得的 embedding）
        self.symbol_embedding = nn.Embedding(symbol_vocab_size, embed_dim)
        self.special_token_embedding = nn.Embedding(2, embed_dim)
        self.non_zero_linear1 = nn.Linear(1, expr_max_bin)
        self.non_zero_silu = nn.SiLU()
        self.non_zero_linear2 = nn.Linear(expr_max_bin, expr_max_bin)
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bin_embedding = nn.Embedding(expr_max_bin, embed_dim)
        
        # gene id embedding原始输出为1024，后续映射到 embed_dim
        self.gene_id_emb = nn.Embedding(symbol_vocab_size, 1024, padding_idx=0)
        self.proj_gene_id = nn.Linear(1024, embed_dim)
        
        self.proj_gene_length = nn.Linear(1, embed_dim)
        self.proj_wiki = nn.Linear(768, embed_dim)
        self.proj_esm2 = nn.Linear(1280, embed_dim)
        self.proj_graph1 = nn.Linear(512, embed_dim)
        self.proj_graph2 = nn.Linear(512, embed_dim)
        self.proj_x = nn.Linear(embed_dim, embed_dim)
        self.proj_gene_id = nn.Linear(1024, embed_dim)

        self.norm_gene = nn.LayerNorm(embed_dim)
        self.norm_wiki = nn.LayerNorm(embed_dim)
        self.norm_esm2 = nn.LayerNorm(embed_dim)
        self.norm_graph1 = nn.LayerNorm(embed_dim)
        self.norm_graph2 = nn.LayerNorm(embed_dim)
        self.norm_x = nn.LayerNorm(embed_dim)
        self.norm_gene_id = nn.LayerNorm(embed_dim)
        
        # 为每个模态定义一个可学习的标量权重
        self.w_main = nn.Parameter(torch.tensor(1.0))
        self.w_gene_id = nn.Parameter(torch.tensor(1.0))
        self.w_gene_length = nn.Parameter(torch.tensor(1.0))
        self.w_wiki = nn.Parameter(torch.tensor(1.0))
        self.w_esm2 = nn.Parameter(torch.tensor(1.0))
        self.w_graph1 = nn.Parameter(torch.tensor(1.0))
        self.w_graph2 = nn.Parameter(torch.tensor(1.0))
        
        # 融合后做一次归一化与激活
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_act = nn.ReLU()
        
        # Transformer 层保持不变
        self.layers = nn.ModuleList([
            BlockMoba(
                embed_dim,
                num_heads,
                n_routed_experts,
                n_activated_experts,
                n_shared_experts,
                moe_inter_dim,
                inter_dim=inter_dim,
                layer_id=i,
                moba_chunk_size=5,
                moba_topk=2
            )
            for i in range(num_layers)
        ])
        
        # 预测头与 cell type 分类保持原来设计
        self.linear_head1 = nn.Linear(2*embed_dim, 1)
        self.cell_type_classifier = nn.Linear(embed_dim, 2894)
        
    def get_expr_embedding(self, expr_val, mask_val):
        zero_mask = (expr_val - 0.0) < 1e-3
        if mask_val is not None:
            mask_val_mask = (expr_val - mask_val) < 1e-3
        else:
            mask_val_mask = torch.zeros_like(expr_val, dtype=torch.bool)
        expr_val_nz_2d = expr_val.unsqueeze(-1)
        alpha_casted = self.alpha.to(expr_val.dtype).to(expr_val.device)
        v1 = self.non_zero_silu(self.non_zero_linear1(expr_val_nz_2d))
        v2 = self.non_zero_linear2(v1) + alpha_casted * expr_val_nz_2d
        v3 = F.softmax(v2, dim=-1)
        E = v3 @ self.bin_embedding.weight
        if zero_mask.any():
            E[zero_mask] = self.special_token_embedding.weight[0].to(E.dtype)
        if mask_val_mask.any():
            E[mask_val_mask] = self.special_token_embedding.weight[1].to(E.dtype)
        return E

    def forward(self, tr_gene_values, tr_gene_ids, expr_val_pooling, gene_id_pooling, *, 
            graph_embs=None, gene_length=None,
            wiki_embedding=None, esm2_embedding=None,   
            new_graph_emb=None, new_graph_emb2=None, mask_x=None,
            new_graph_gene_ids=None, new_graph_gene_ids2=None, mask_val=-1.0):
        # 主模态 embedding
        emb1 = self.get_expr_embedding(tr_gene_values, mask_val=mask_val)
        x = emb1

        # 模态降维与融合：必须保证所有需要融合的模态都不为空
        if all(v is not None for v in [gene_length, new_graph_emb, new_graph_emb2, wiki_embedding, esm2_embedding]):
            # 投影各个模态并分别归一化
            gene_feat = self.norm_gene(self.proj_gene_length(gene_length.unsqueeze(-1)))
            wiki_feat   = self.norm_wiki(self.proj_wiki(wiki_embedding))
            esm2_feat   = self.norm_esm2(self.proj_esm2(esm2_embedding))
            graph1_feat = self.norm_graph1(self.proj_graph1(new_graph_emb))
            graph2_feat = self.norm_graph2(self.proj_graph2(new_graph_emb2))
            main_x      = self.norm_x(self.proj_x(x))
            
            if tr_gene_ids is not None:
                gene_id_emb = self.norm_gene_id(self.proj_gene_id(self.gene_id_emb(tr_gene_ids)))
            else:
                gene_id_emb = torch.zeros_like(main_x)
            
            # 融合前各模态都已经归一化，直接对各自乘以对应的可训练权重相加
            fused = (self.w_main      * main_x +
                    self.w_gene_id   * gene_id_emb +
                    self.w_gene_length * gene_feat +
                    self.w_wiki      * wiki_feat +
                    self.w_esm2      * esm2_feat +
                    self.w_graph1    * graph1_feat +
                    self.w_graph2    * graph2_feat)
                    
            # 融合后做一次归一化与激活
            fused = self.fusion_act(self.fusion_norm(fused))
            
            # 更新 x 为融合后的结果
            x = fused

        for layer in self.layers:
            x = layer(x, cross_input=None, mask=mask_x)

        # 以下部分保持原有逻辑
        cell_embedding = x[:, 0, :]  # 取 CLS token
        cell_type_logits = self.cell_type_classifier(cell_embedding)
        cell_embedding_expanded = cell_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        # 如果有额外的图信息，则与 CLS token 进行简单相加
        if graph_embs is not None:
            cell_embedding_expanded = cell_embedding_expanded + graph_embs
        x = torch.cat([x, cell_embedding_expanded], dim=-1)
        x1_output = self.linear_head1(x)
        return x1_output, cell_type_logits

class SequenceDataset(Dataset):
    def __init__(self, seq_length, vocab_size, num_samples):
        """
        Initialize the dataset with sequences of token indices.
        :param seq_length: The length of each input sequence.
        :param vocab_size: The size of the vocabulary.
        :param num_samples: The number of samples in the dataset.
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        # Generate random sequences of token indices
        self.data = [torch.randint(0, vocab_size, (seq_length,)) for _ in range(num_samples)]

        # Generate random labels for binary classification
        self.labels = [random.randint(0, 1) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return a single data sample (input sequence and label).
        :param idx: Index of the sample.
        :return: Tuple (input_sequence, label).
        """
        return self.data[idx], self.labels[idx]


class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size, embed_dim):
        """
        Create a dummy dataset for testing.
        Each sample contains:
          - sp_mask_values: [seq_length] integers in [0, vocab_size)
          - sp_mask_gene_ids: [seq_length] integers in [0, vocab_size)
          - sc_mask_values: [seq_length] integers in [0, vocab_size)
          - sc_mask_gene_ids: [seq_length] integers in [0, vocab_size)
          - graph_input: [seq_length, embed_dim] float tensor
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sp_mask_values = torch.randint(0, self.vocab_size, (self.seq_length,))
        sp_mask_gene_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        sc_mask_values = torch.randint(0, self.vocab_size, (self.seq_length,))
        sc_mask_gene_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        # Create a random embedding vector and repeat it along the sequence dimension.
        graph_emb = torch.randn(self.embed_dim)
        graph_input = graph_emb.unsqueeze(0).repeat(self.seq_length, 1)
        return {
            'sp_mask_values': sp_mask_values,
            'sp_mask_gene_ids': sp_mask_gene_ids,
            'sc_mask_values': sc_mask_values,
            'sc_mask_gene_ids': sc_mask_gene_ids,
            'graph_input': graph_input
        }

# ------------------------------
# Dataloader Creation Function
# ------------------------------
def create_dummy_dataloader(batch_size, num_samples, seq_length, vocab_size, embed_dim, is_ddp):
    dataset = DummyDataset(num_samples, seq_length, vocab_size, embed_dim)

    # output the size of dataset
    print(f"Dataset size: {len(dataset)}")

    if is_ddp:
        sampler = DistributedSampler(dataset)
        shuffle = False  # Sampler will handle shuffling.
    else:
        sampler = None
        shuffle = True
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,  # Set to >0 if desired.
        pin_memory=True
    )
    return dataloader


def setup_ddp():
    """
    Initialize the distributed process group and set the local device.
    """
    # Initialize using environment variables (set by torchrun)
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """
    Destroy the distributed process group.
    """
    dist.destroy_process_group()

def test_deepseek2():
    # Define the model parameters
    num_layers = 2
    embed_dim = 512
    num_heads = 2
    n_routed_experts = 3
    n_activated_experts = 2
    n_shared_experts = 1
    moe_inter_dim = 4
    seq_length = 10
    vocab_size = 100

    # Initialize the model
    model = TransformerDeepSeek2(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        n_routed_experts=n_routed_experts,
        n_activated_experts=n_activated_experts,
        n_shared_experts=n_shared_experts,
        moe_inter_dim=moe_inter_dim,
        seq_length=seq_length,
        vocab_size=vocab_size
    )

    # Get the local rank and set device accordingly
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)

    # Move model to device and wrap with DDP
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)
    model.train()

    # Create the dummy dataloader. For testing, we create 20 samples.
    batch_size = 2
    num_samples = 20
    is_ddp = True  # Set to True since we are running under DDP.
    dataloader = create_dummy_dataloader(batch_size, num_samples, seq_length, vocab_size, embed_dim, is_ddp)

    # Define CrossEntropyLoss
    # criterion = nn.CrossEntropyLoss()
    #define huber loss
    criterion = nn.HuberLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Iterate through one epoch of data and run the forward pass.
    for batch_idx, batch in enumerate(dataloader):
        print('batch_idx', batch_idx)

        # Ensure all tensors are on the correct device.
        sp_mask_values = batch['sp_mask_values'].to(device)
        sp_mask_gene_ids = batch['sp_mask_gene_ids'].to(device)
        sc_mask_values = batch['sc_mask_values'].to(device)
        sc_mask_gene_ids = batch['sc_mask_gene_ids'].to(device)
        graph_input = batch['graph_input'].to(device)

        # Forward pass through the model
        output1, output2 = model(sp_mask_values, sp_mask_gene_ids, sc_mask_values, sc_mask_gene_ids, graph_input)

        loss = criterion(output1.view(-1, output1.size(-1)), sp_mask_gene_ids.view(-1))  # Flatten for CrossEntropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # For demonstration, only rank 0 prints the outputs.
        if dist.get_rank() == 0:
            print("Output1 shape:", output1.shape)
            print("Output2 shape:", output2.shape)
        # break  # Run one batch for testing.


if __name__ == '__main__':
    # Setup distributed environment
    local_rank = setup_ddp()

    test_deepseek2()

    # Cleanup when done
    cleanup_ddp()

    # # Parameters
    # seq_length = 50   # Length of sequences
    # vocab_size = 10 # Vocabulary size
    # num_samples = 1000 # Number of samples
    #
    # # Create dataset and dataloader
    # dataset = SequenceDataset(seq_length, vocab_size, num_samples)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #
    # # Inspect a single batch
    # for inputs, labels in dataloader:
    #     print("Input Shape:", inputs.shape)  # (batch_size, seq_length)
    #     print("Labels Shape:", labels.shape)  # (batch_size,)
    #     break
    #
    # print('inputs', inputs)
    #
    # transformer = TransformerDeepSeek(
    #     num_layers=2,
    #     embed_dim=64,
    #     num_heads=2,
    #     n_routed_experts=3,
    #     n_activated_experts=2,
    #     n_shared_experts=1,
    #     moe_inter_dim=32,
    #     seq_length=seq_length
    # )
    #
    #
    #
    # # Define optimizer and loss function
    # optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3)
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # # Training loop
    # for epoch in range(5):  # Train for 5 epochs
    #     for inputs, labels in dataloader:
    #         optimizer.zero_grad()
    #
    #         # Forward pass
    #         cross_input = nn.Embedding(seq_length, 64)(inputs)
    #         # print('the shape of cross input', cross_input.shape)
    #         outputs = transformer(inputs,cross_input)
    #         logits = outputs.mean(dim=1)  # Pooling for classification
    #         loss = criterion(logits, labels)
    #
    #         # Backward pass
    #         loss.backward()
    #         optimizer.step()
    #
    #     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    #
