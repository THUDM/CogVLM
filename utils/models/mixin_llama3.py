import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.transformer_defaults import attention_fn_default
from sat.model.base_model import BaseModel, BaseMixin, non_conflict
from sat.model import ChatGLMModel
from sat.model.position_embedding.rotary_embeddings import RotaryEmbedding, apply_rotary_pos_emb_index
from sat.mpu.layers import ColumnParallelLinear, RowParallelLinear
from sat.mpu.utils import split_tensor_along_last_dim, gelu
from sat import mpu
from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding


class LlamaVisionExpertFCMixin(BaseMixin):
    def __init__(self, in_features, hidden_features, num_layers=32, num_vision_layers=0, dropout=0.,
                 use_lora=False, lora_rank=4, lora_alpha=1, lora_dropout=0., frequency=1,
                 params_dtype=torch.float, device=torch.device('cpu')):
        super().__init__()

        self.num_layers = num_layers
        self.num_vision_layers = num_vision_layers
        vision_layer_range = [i for i in range(0, min(num_vision_layers, num_layers), frequency)]
        self.vision_layer_range = vision_layer_range
        self.gate_proj = nn.ModuleList([ColumnParallelLinear(
            in_features,
            hidden_features,
            gather_output=False,
            init_method=None,
            bias=False,
            params_dtype=params_dtype,
            module=self,
            name="dense_h_to_4h_gate",
            skip_init=True,
            device=device
        ) for i in range(num_layers)])

        if dropout > 0:
            self.dropout = nn.Dropout(p=lora_dropout)
        else:
            self.dropout = lambda x: x
        self.use_lora = use_lora
        # Trainable vision expert parameters
        vision_dense_h_to_4h_list = []
        vision_dense_4h_to_h_list = []
        gate_proj_list = []


        for i in vision_layer_range:
            vision_dense_h_to_4h = ColumnParallelLinear(
                in_features,
                hidden_features,
                gather_output=False,
                init_method=None,
                bias=False,
                params_dtype=params_dtype,
                module=self,
                name="vision_dense_h_to_4h",
                skip_init=True,
                device=device
            )

            # Project back to h.
            vision_dense_4h_to_h = RowParallelLinear(
                hidden_features,
                in_features,
                input_is_parallel=True,
                init_method=None,
                bias=False,
                params_dtype=params_dtype,
                module=self,
                name="vision_dense_4h_to_h",
                skip_init=True,
                device=device
            )

            gate_proj = ColumnParallelLinear(
                in_features,
                hidden_features,
                gather_output=False,
                init_method=None,
                bias=False,
                params_dtype=params_dtype,
                module=self,
                name="vision_gate_proj",
                skip_init=True,
                device=device
            )

            # nn.init.kaiming_uniform_(vision_dense_h_to_4h.weight)
            # nn.init.kaiming_uniform_(vision_dense_4h_to_h.weight)
            # nn.init.kaiming_uniform_(gate_proj.weight)


            vision_dense_h_to_4h_list.append(vision_dense_h_to_4h)
            vision_dense_4h_to_h_list.append(vision_dense_4h_to_h)
            gate_proj_list.append(gate_proj)

        self.vision_dense_h_to_4h_list = nn.ModuleDict([
            (str(layer_id), vision_dense_h_to_4h)
            for layer_id, vision_dense_h_to_4h in zip(vision_layer_range, vision_dense_h_to_4h_list)
        ])
        self.vision_dense_4h_to_h_list = nn.ModuleDict([
            (str(layer_id), vision_dense_4h_to_h)
            for layer_id, vision_dense_4h_to_h in zip(vision_layer_range, vision_dense_4h_to_h_list)
        ])
        self.vision_gate_proj = nn.ModuleDict([
            (str(layer_id), gate_proj)
            for layer_id, gate_proj in zip(vision_layer_range, gate_proj_list)
        ])

        if self.use_lora:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            if lora_dropout and lora_dropout > 0:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout = lambda x: x

            self.lora_linear = nn.ModuleDict([
                (str(layer_id), nn.ParameterDict())
                for layer_id in vision_layer_range
            ])

            for i in vision_layer_range:
                i = str(i)
                self.lora_linear[i]["vision_gate_proj" + "_A"] = nn.Parameter(torch.zeros((lora_rank, in_features)))
                self.lora_linear[i]["vision_gate_proj" + "_B"] = nn.Parameter(torch.zeros((hidden_features, lora_rank)))
                nn.init.kaiming_uniform_(self.lora_linear[i]["vision_gate_proj" + "_A"], a=math.sqrt(5))
                nn.init.zeros_(self.lora_linear[i]["vision_gate_proj" + "_B"])

                self.lora_linear[i]["vision_dense_h_to_4h" + "_A"] = nn.Parameter(torch.zeros((lora_rank, in_features)))
                self.lora_linear[i]["vision_dense_h_to_4h" + "_B"] = nn.Parameter(torch.zeros((hidden_features, lora_rank)))
                nn.init.kaiming_uniform_(self.lora_linear[i]["vision_dense_h_to_4h" + "_A"], a=math.sqrt(5))
                nn.init.zeros_(self.lora_linear[i]["vision_dense_h_to_4h" + "_B"])

                self.lora_linear[i]["vision_dense_4h_to_h" + "_A"] = nn.Parameter(torch.zeros((lora_rank, in_features)))
                self.lora_linear[i]["vision_dense_4h_to_h" + "_B"] = nn.Parameter(torch.zeros((hidden_features, lora_rank)))
                nn.init.kaiming_uniform_(self.lora_linear[i]["vision_dense_4h_to_h" + "_A"], a=math.sqrt(5))
                nn.init.zeros_(self.lora_linear[i]["vision_dense_4h_to_h" + "_B"])

            self.scaling = self.lora_alpha / self.lora_rank

    def mlp_forward(self, hidden_states, **kw_args):
        mixin_self = self
        self = self.transformer.layers[kw_args['layer_id']].mlp
        if "vision_expert_mask" in kw_args:
            vision_expert_mask = kw_args['vision_expert_mask']
        else:
            vision_expert_mask = None

        layer_id_key = str(int(kw_args['layer_id']))

        if kw_args['layer_id'] in mixin_self.vision_layer_range and (vision_expert_mask is not None) and vision_expert_mask.any():
            vision_dense_h_to_4h = mixin_self.vision_dense_h_to_4h_list[layer_id_key]
            vision_dense_4h_to_h = mixin_self.vision_dense_4h_to_h_list[layer_id_key]
            vision_gate_proj = mixin_self.vision_gate_proj[layer_id_key]
            output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)

            language_hidden_state = hidden_states[~vision_expert_mask.bool()].contiguous()
            language_intermediate_parallel = self.activation_func(mixin_self.gate_proj[kw_args['layer_id']](language_hidden_state)) * self.dense_h_to_4h(language_hidden_state)
            output[~vision_expert_mask.bool()] = self.dense_4h_to_h(language_intermediate_parallel)  # language_output

            vision_hidden_state = hidden_states[vision_expert_mask.bool()].contiguous()
            vision_intermediate_parallel = vision_dense_h_to_4h(vision_hidden_state)
            gate_output = vision_gate_proj(vision_hidden_state)

            if mixin_self.use_lora:
                lora_layer = mixin_self.lora_linear[layer_id_key]
                gate_output += (mixin_self.lora_dropout(vision_hidden_state)) @ lora_layer["vision_gate_proj_A"].T @ lora_layer["vision_gate_proj_B"].T * mixin_self.scaling
                vision_intermediate_parallel += (mixin_self.lora_dropout(vision_hidden_state) @ lora_layer["vision_dense_h_to_4h_A"].T @ lora_layer["vision_dense_h_to_4h_B"].T) * mixin_self.scaling
                vision_intermediate_parallel *= self.activation_func(gate_output)
                output[vision_expert_mask.bool()] = vision_dense_4h_to_h(vision_intermediate_parallel) + \
                         (mixin_self.lora_dropout(vision_intermediate_parallel) @ lora_layer["vision_dense_4h_to_h_A"].T @ lora_layer["vision_dense_4h_to_h_B"].T) * mixin_self.scaling
            else:
                vision_intermediate_parallel *= self.activation_func(gate_output)
                output[vision_expert_mask.bool()] = vision_dense_4h_to_h(vision_intermediate_parallel)  # vision_output
        else:
            intermediate_parallel = self.activation_func(mixin_self.gate_proj[kw_args['layer_id']](hidden_states)) * self.dense_h_to_4h(hidden_states)
            output = self.dense_4h_to_h(intermediate_parallel)

        return output.contiguous()

    def copy_param(self):
        with torch.no_grad():
            for i in self.vision_layer_range:
                self.vision_gate_proj[str(i)].weight.data.copy_(self.gate_proj[i].weight.data)
                self.vision_dense_4h_to_h_list[str(i)].weight.data.copy_(self.transformer.layers[i].mlp.dense_4h_to_h.weight.data)
                self.vision_dense_h_to_4h_list[str(i)].weight.data.copy_(self.transformer.layers[i].mlp.dense_h_to_4h.weight.data)

from sat.mpu import get_model_parallel_world_size
from sat.mpu.utils import divide
from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding

class LlamaVisionExpertAttnMixin(BaseMixin):
    def __init__(self, hidden_size, num_attention_heads, num_layers=28, num_vision_layers=0, use_vision_expert=True,
                 use_lora=False, lora_rank=4, lora_alpha=1, lora_dropout=0., num_multi_query_heads=0, frequency=1,
                 params_dtype=torch.float, device=torch.device('cpu')):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_multi_query_heads = num_multi_query_heads
        world_size = get_model_parallel_world_size()

        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.num_multi_query_heads_per_partition = divide(num_multi_query_heads, world_size)
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition

        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head

        self.rotary_emb = FastRotaryEmbedding(
            hidden_size // num_attention_heads, pos_idx_in_fp32=False, base=500000
        )

        self.num_vision_layers = num_vision_layers
        self.num_layers = num_layers
        vision_layer_range = [i for i in range(0, min(num_vision_layers, num_layers), frequency)]
        self.vision_layer_range = vision_layer_range

        self.use_lora = use_lora
        self.use_vision_expert = use_vision_expert
        # Trainable vision expert parameters

        if num_multi_query_heads == 0:
            self.qkv_size = 3 * self.hidden_size
            self.stride = 3
        else:  # multi-query
            self.qkv_size = self.hidden_size + self.hidden_size_per_attention_head * self.num_multi_query_heads * 2
            self.stride = [self.num_attention_heads_per_partition, self.num_multi_query_heads_per_partition,
                           self.num_multi_query_heads_per_partition]

        if self.use_vision_expert:
            vision_query_key_value_list = []
            vision_dense_list = []
            for i in vision_layer_range:
                vision_query_key_value = ColumnParallelLinear(
                    self.hidden_size,
                    self.qkv_size,
                    stride=self.stride,
                    gather_output=False,
                    init_method=None,
                    bias=True,
                    params_dtype=params_dtype,
                    module=self,
                    name="vision_query_key_value",
                    skip_init=True,
                    device=device
                )

                vision_dense = RowParallelLinear(
                    self.inner_hidden_size,
                    hidden_size,
                    input_is_parallel=True,
                    init_method=None,
                    bias=False,
                    params_dtype=params_dtype,
                    module=self,
                    name="vision_dense",
                    skip_init=True,
                    device=device,
                    final_bias=False
                )

                # nn.init.kaiming_uniform_(vision_query_key_value.weight)
                # nn.init.kaiming_uniform_(vision_dense.weight)

                vision_query_key_value_list.append(vision_query_key_value)
                vision_dense_list.append(vision_dense)

            self.vision_query_key_value_list = nn.ModuleDict([
                (str(layer_id), vision_query_key_value)
                for layer_id, vision_query_key_value in zip(vision_layer_range, vision_query_key_value_list)
            ])
            self.vision_dense_list = nn.ModuleDict([
                (str(layer_id), vision_dense)
                for layer_id, vision_dense in zip(vision_layer_range, vision_dense_list)
            ])

        if self.use_lora:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout
            if lora_dropout and lora_dropout > 0:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout = lambda x: x

            self.lora_linear = nn.ModuleDict([
                (str(layer_id), nn.ParameterDict())
                for layer_id in vision_layer_range
            ])
            matrices = ["Q", "K", "V", "O"]

            for i in vision_layer_range:
                i = str(i)
                for matrix in matrices:
                    self.lora_linear[i][matrix + "_A"] = nn.Parameter(torch.zeros((lora_rank, hidden_size)))
                    self.lora_linear[i][matrix + "_B"] = nn.Parameter(torch.zeros((hidden_size, lora_rank)))
                    nn.init.kaiming_uniform_(self.lora_linear[i][matrix + "_A"], a=math.sqrt(5))
                    nn.init.zeros_(self.lora_linear[i][matrix + "_B"])

            self.scaling = self.lora_alpha / self.lora_rank

    def attention_forward(self, hidden_states, mask, **kw_args):
        mixin_self = self
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = attention_fn_default
        if 'attention_fn' in self.hooks:
            attention_fn = self.hooks['attention_fn']
        if "vision_expert_mask" in kw_args:
            vision_expert_mask = kw_args['vision_expert_mask']
        else:
            vision_expert_mask = None

        layer_id_key = str(int(kw_args['layer_id']))
        if mixin_self.use_vision_expert and kw_args['layer_id'] in mixin_self.vision_layer_range and vision_expert_mask is not None:
            shape = list(hidden_states.shape)
            shape[-1] = mixin_self.qkv_size # different from llama: multi query
            parallel_size = mpu.get_model_parallel_world_size()
            shape[-1] = shape[-1] // parallel_size
            vision_query_key_value = mixin_self.vision_query_key_value_list[layer_id_key]
            mixed_raw_layer = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
            language_hidden_states = hidden_states[~vision_expert_mask.bool()]
            vision_hidden_states = hidden_states[vision_expert_mask.bool()]
            mixed_raw_layer[~vision_expert_mask.bool()] = self.query_key_value(
                language_hidden_states)  # language_mixed_raw_layer
            mixed_raw_layer[vision_expert_mask.bool()] = vision_query_key_value(
                vision_hidden_states)  # vision_mixed_raw_layer
        else:
            mixed_raw_layer = self.query_key_value(hidden_states)

        if mixin_self.use_lora and kw_args['layer_id'] in mixin_self.vision_layer_range and (
                vision_expert_mask is not None) and vision_expert_mask.any():
            lora_layer = mixin_self.lora_linear[layer_id_key]
            (mixed_query_layer, mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

            mixed_query_layer[vision_expert_mask.bool()] += (mixin_self.lora_dropout(
                vision_hidden_states) @ lora_layer["Q_A"].T @ lora_layer[
                                                                 "Q_B"].T) * mixin_self.scaling
            mixed_key_layer[vision_expert_mask.bool()] += (mixin_self.lora_dropout(
                vision_hidden_states) @ lora_layer["K_A"].T @ lora_layer[
                                                               "K_B"].T) * mixin_self.scaling
            mixed_value_layer[vision_expert_mask.bool()] += (mixin_self.lora_dropout(
                vision_hidden_states) @ lora_layer["V_A"].T @ lora_layer[
                                                                 "V_B"].T) * mixin_self.scaling
        else:
            (mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, self.stride)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer).contiguous()
        key_layer = self._transpose_for_scores(mixed_key_layer).contiguous()
        value_layer = self._transpose_for_scores(mixed_value_layer).contiguous()

        # cos, sin = mixin_self.rotary_emb(value_layer, seq_len=kw_args['position_ids'].max()+1)
        # query_layer, key_layer = apply_rotary_pos_emb_index_bhs(query_layer, key_layer, cos, sin, kw_args['position_ids'])

        query_layer, key_layer = mixin_self.rotary_emb(query_layer,key_layer, kw_args['position_ids'], max_seqlen=kw_args['position_ids'].max()+1, layer_id=kw_args['layer_id'])


        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if mixin_self.use_vision_expert and kw_args['layer_id'] in mixin_self.vision_layer_range and (
                vision_expert_mask is not None) and vision_expert_mask.any():
            vision_dense = mixin_self.vision_dense_list[layer_id_key]
            parallel_size = mpu.get_model_parallel_world_size()
            target_shape = context_layer.shape[:-1] + (context_layer.shape[-1] * parallel_size,)
            output = torch.empty(target_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            output[~vision_expert_mask.bool()] = self.dense(context_layer[~vision_expert_mask.bool()].contiguous())  # language
            output[vision_expert_mask.bool()] = vision_dense(context_layer[vision_expert_mask.bool()].contiguous())  # vision
        else:
            output = self.dense(context_layer)

        if mixin_self.use_lora and kw_args['layer_id'] in mixin_self.vision_layer_range and (
                vision_expert_mask is not None) and vision_expert_mask.any():
            lora_layer = mixin_self.lora_linear[layer_id_key]
            output[vision_expert_mask.bool()] += (mixin_self.lora_dropout(context_layer[vision_expert_mask.bool()]) @
                                                  lora_layer["O_A"].T @ lora_layer["O_B"].T) * mixin_self.scaling

        if self.training:
            output = self.output_dropout(output)
        return output.contiguous()

    def copy_param(self):
        with torch.no_grad():
            for i in self.vision_layer_range:
                self.vision_query_key_value_list[str(i)].weight.data.copy_(self.transformer.layers[i].attention.query_key_value.weight.data)
                self.vision_dense_list[str(i)].weight.data.copy_(self.transformer.layers[i].attention.dense.weight.data)