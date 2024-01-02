"""largely copy from llama and adapt for CogAgent"""
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Literal, Dict, Any

import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from einops import rearrange

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils.logging import get_logger
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_cogagent import CogAgentConfig
# from .util import FastRotaryEmbedding
from torch.nn import functional as F
from .visual import EVA2CLIPModel
from .cross_visual import CrossVisionModel

if TYPE_CHECKING:
    from transformers.utils import ModelOutput

logger = get_logger(__name__)

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def get_expert_mask(token_type_ids: "torch.LongTensor(B, L)") -> "[torch.BoolTensor(B, L), torch.BoolTensor(B, L)]":
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:] == VISION_TOKEN_TYPE)
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class VisionExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_mlp = MLP(config)
        self.vision_mlp = MLP(config)

    def forward(self, hidden_states: "torch.Tensor(B, L, D)", token_type_ids: "torch.LongTensor(B, L)"):
        output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


def attention_fn(
        query_layer: "torch.tensor(B, H, L, HD)",
        key_layer: "torch.tensor(B, H, L, HD)",
        value_layer: "torch.tensor(B, H, L, HD)",
        attention_mask: "torch.tensor(B, H, L, HD)",
        *,
        scaling_attention_score: bool = True,
        attention_dropout: nn.Module = None
):
    attention_mask_bool = (attention_mask == 0)
    is_low_triangle = (attention_mask_bool == torch.ones_like(attention_mask_bool, dtype=torch.float).tril()).all()
    is_full = (attention_mask_bool > 0).all()
    if not (int(torch.__version__.split('.')[0]) >= 2):
        warnings.warn("It's recommended to use torch2.0 or higher.")
    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        return torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=not is_full
        )
    else:
        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask
        attention_scores = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        if attention_dropout is not None:
            attention_scores = attention_dropout(attention_scores)
        context_layer = torch.matmul(attention_scores, value_layer)
        return context_layer

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = 0

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, device=device) / self.dim)
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[:, None, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[:, None, :].to(dtype), persistent=False)

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id):
    # batch_size, num_head, seq_len, hidden_size
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(1)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k

class VisionExpertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.vision_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.vision_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.language_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.language_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            token_type_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)

        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        mixed_raw_layer = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])

        query_states, key_states, value_states = torch.split(mixed_raw_layer, self.hidden_size, dim=-1)
        query_states = self._transpose_for_scores(query_states)  # B, H, L, HD
        key_states = self._transpose_for_scores(key_states)  # B, H, L, HD
        value_states = self._transpose_for_scores(value_states)  # B, H, L, HD

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)
        query_states, key_states = apply_rotary_pos_emb_index_bhs(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        context_layer = attention_fn(
            query_layer=query_states, key_layer=key_states, value_layer=value_states, attention_mask=attention_mask,
            scaling_attention_score=True, attention_dropout=None)
        if context_layer.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.size()}"
            )
        context_layer = context_layer.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

        attn_output = torch.empty(context_layer.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])

        if output_attentions:
            warnings.warn("output_attentions is not implemented.")

        return attn_output, None, past_key_value

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.cross_hidden_size = config.cross_hidden_size
        self.cross_compute_hidden_size = config.cross_compute_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.cross_head_dim = self.cross_compute_hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.query = nn.Linear(self.hidden_size, self.cross_compute_hidden_size, bias=False)
        self.key_value = nn.Linear(self.cross_hidden_size, self.cross_compute_hidden_size * 2, bias=False)
        self.dense = nn.Linear(self.cross_compute_hidden_size, self.hidden_size, bias=False)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.size()[:-1] + (self.num_heads, self.cross_head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_outputs: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3

        mixed_query_layer = self.query(hidden_states)
        if past_key_value is None:
            mixed_x_layer = self.key_value(encoder_outputs)
            mixed_key_layer, mixed_value_layer = torch.split(mixed_x_layer, self.cross_compute_hidden_size, dim=-1)
            key_states = self._transpose_for_scores(mixed_key_layer)  # B, H, L, HD
            value_states = self._transpose_for_scores(mixed_value_layer)  # B, H, L, HD
        else:
            key_states, value_states = past_key_value

        query_states = self._transpose_for_scores(mixed_query_layer)  # B, H, L, HD

        past_key_value = (key_states, value_states) if use_cache else None

        context_layer = attention_fn(
            query_layer=query_states, key_layer=key_states, value_layer=value_states, attention_mask=attention_mask,
            scaling_attention_score=True, attention_dropout=None)
        if context_layer.size() != (bsz, self.num_heads, q_len, self.cross_head_dim):
            raise ValueError(
                f"`cross_attn_output` should be of size {(bsz, self.num_heads, q_len, self.cross_head_dim)}, but is"
                f" {context_layer.size()}"
            )
        context_layer = context_layer.transpose(1, 2).contiguous().reshape(bsz, q_len, self.cross_hidden_size)

        attn_output = self.dense(context_layer)

        if output_attentions:
            warnings.warn("output_attentions is not implemented.")

        return attn_output, None, past_key_value

class CogAgentDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VisionExpertAttention(config=config)
        self.cross_attn = CrossAttention(config=config)
        self.mlp = VisionExpertMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_outputs: torch.Tensor,
            token_type_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value[:2] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        cross_input = self.post_cross_attention_layernorm(hidden_states)
        # Fully Connected
        attention_output, self_cross_attn_weights, present_cross_key_value = self.cross_attn(
            hidden_states=cross_input,
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attention_mask,
            past_key_value=past_key_value[-2:] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            )
        hidden_states = hidden_states + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(mlp_input, token_type_ids=token_type_ids)
        hidden_states = mlp_output + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value+present_cross_key_value,)

        return outputs  # type: ignore


class CogAgentPreTrainedModel(PreTrainedModel):
    config_class = CogAgentConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CogAgentDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if len(image_list):
            return False
    return True


def build_position_ids(x: "torch.BoolTensor(B, L)", attention_mask: Optional["torch.BoolTensor(B, L)"] = None) -> "torch.LongTensor(B, L)":
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | ((tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(dim=-1)
    return y


class CogAgentModel(CogAgentPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([CogAgentDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision = EVA2CLIPModel(config)
        self.cross_vision = CrossVisionModel(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def encode_images(self, images: List[List[torch.Tensor]]) -> torch.Tensor:
        images_list, images = images, []

        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)

        images = torch.stack(images)
        images_features = self.vision(images)
        return images_features

    def encode_cross_images(self, images: List[List[torch.Tensor]]) -> torch.Tensor:
        images_list, images = images, []

        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)

        images = torch.stack(images)
        encoder_outputs = self.cross_vision(images)
        return encoder_outputs
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            images: List[List[torch.Tensor]] = None,
            cross_images: List[List[torch.Tensor]] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""

        if past_key_values is not None:
            encoder_outputs = None
            # generate mode with past_key_values. the image features are already mapped
        else:
            # not allow for inputs_embeds, because we want to process image feature
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            if not is_empty(images):  # multi-modality
                assert token_type_ids is not None, f"multi-modality requires `token_type_ids`!"
                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
                inputs_embeds = self.embed_tokens(input_ids)
                images_features = self.encode_images(images)
                encoder_outputs = self.encode_cross_images(cross_images)
                images_features = rearrange(images_features, 'b n d -> (b n) d')
                images_features = images_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)
            else:  # single-modality
                if token_type_ids is None:
                    token_type_ids = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device) * LANGUAGE_TOKEN_TYPE
                assert not (token_type_ids == VISION_TOKEN_TYPE).any(), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)
                encoder_outputs = None

            if position_ids is None:
                position_ids = build_position_ids(token_type_ids, attention_mask)
            input_ids = None

        return self.llm_forward(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
            self,
            input_ids: torch.LongTensor = None,
            encoder_outputs: torch.LongTensor = None,
            token_type_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for CogAgent with `token_type_ids`"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        if cross_attention_mask is None:
            cross_attention_mask = torch.zeros(
                (batch_size, 1, 1, 1), dtype=attention_mask.dtype, device=inputs_embeds.device
            ) # 0 or -inf
        

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                encoder_outputs=encoder_outputs,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # noinspection PyMethodMayBeStatic
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

def vqa_history_to_prompt(history, query):
    # Only support single round chat in vqa mode
    prompt = "<EOI>Question: "
    # for i, (old_query, response) in enumerate(history):
    #     prompt += old_query + " Short answer: " + response + " Question: "
    prompt += query + " Short answer:"
    return prompt

def chat_old_history_to_prompt(history, query):
    prompt = "<EOI>Question: "
    for i, (old_query, response) in enumerate(history):
        prompt += old_query + " Answer: " + response + "\nQuestion: "
    prompt += query + " Answer:"
    return prompt

def chat_history_to_prompt(history, query):
    prompt = " [INST] "
    for i, (old_query, response) in enumerate(history):
        prompt += old_query + " [/INST] " + response + " [INST] "
    prompt += query + " [/INST] "
    return prompt


def base_history_to_prompt(history, query):
    prompt = query
    return prompt


_history_to_prompt = {
    "base": base_history_to_prompt,
    "chat": chat_history_to_prompt,
    "chat_old": chat_old_history_to_prompt,
    "vqa": vqa_history_to_prompt
}


class CogAgentForCausalLM(CogAgentPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)
        self.model = CogAgentModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            images: List[List[torch.Tensor]] = None,
            cross_images: List[List[torch.Tensor]] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            cross_images=cross_images,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_attention_mask_for_generation(
            self,
            inputs: torch.Tensor,
            pad_token_id: Optional[int],
            eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)  # type: ignore

    def prepare_inputs_for_generation(
            self, input_ids, token_type_ids, images=None, cross_images=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # build position_ids if needed
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = build_position_ids(token_type_ids, attention_mask)

        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "token_type_ids": token_type_ids,
                "images": images,
                "cross_images": cross_images,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
            self,
            outputs: "ModelOutput",
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            new_token_type_ids = torch.ones(size=(token_type_ids.shape[0], 1), dtype=token_type_ids.dtype, device=token_type_ids.device) * LANGUAGE_TOKEN_TYPE
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, new_token_type_ids], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def build_conversation_input_ids(
            self,
            tokenizer: "PreTrainedTokenizer",
            *,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List["PIL.Image"]] = None,
            template_version: Optional[Literal["base", "chat", "vqa"]] = None,
            answer: str = None,
    ):
        image_size: int = self.config.vision_config['image_size']
        cross_image_size: int = self.config.cross_image_size
        patch_size: int = self.config.vision_config['patch_size']
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []
        text = _history_to_prompt[template_version](history, query)

        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            ori = images
            # vision
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
            images = [transform(ori[0])]
            cross_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (cross_image_size, cross_image_size), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
            cross_images = [cross_transform(ori[0])]
            # language
            vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if answer is not None:
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            answer_ids += [tokenizer.eos_token_id]
            text_ids += answer_ids

        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)
        
        if answer is not None:
            labels = [-100 for _ in range(len(input_ids)-len(answer_ids))] + answer_ids
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = None
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'images': images,
            'cross_images': cross_images,
            'labels': labels,
        }
