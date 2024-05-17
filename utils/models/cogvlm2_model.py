from sat.model.official.llama_model import LLaMAModel
import json
import torch
from sat.model.base_model import BaseMixin
import torch.nn as nn
from .mixin_llama3 import LlamaVisionExpertFCMixin, LlamaVisionExpertAttnMixin

from sat.resources.urls import MODEL_URLS

# MODEL_URLS["cogvlm-base-224"] = "r2://cogvlm-base-224.zip"
# MODEL_URLS["cogvlm-base-490"] = "r2://cogvlm-base-490.zip"
# MODEL_URLS["cogvlm-chat-v1.1"] = "r2://cogvlm-chat-v1.1.zip"
# MODEL_URLS["cogvlm-grounding-base"] = "r2://cogvlm-grounding-base.zip"
# MODEL_URLS["cogvlm-grounding-generalist-v1.1"] = "r2://cogvlm-grounding-generalist-v1.1.zip"


class GLU(nn.Module):
    def __init__(self, args, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, args.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(args.inner_hidden_size, args.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x

from .eva_clip_model import EVA2CLIPModel
import argparse
from copy import deepcopy
def override_dist_dtype_device_args(args, b={}):
    if args.mode == 'inference':
        minimal_args = argparse.Namespace(
            world_size=args.world_size,
            rank=args.rank,
            local_rank=args.local_rank,
            skip_init=args.skip_init,
            use_gpu_initialization=args.use_gpu_initialization,
            deepspeed=args.deepspeed,
            bf16=args.bf16,
            fp16=args.fp16,
            mode=args.mode,
            device=args.device
        )
    else:
        minimal_args = argparse.Namespace(
                world_size=args.world_size,
                rank=args.rank,
                local_rank=args.local_rank,
                skip_init=args.skip_init,
                use_gpu_initialization=args.use_gpu_initialization,
                deepspeed=args.deepspeed,
                bf16=args.bf16,
                fp16=args.fp16,
                mode=args.mode,
                checkpoint_activations=args.checkpoint_activations if not hasattr(args, 'vit_checkpoint_activations') else args.vit_checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                device=args.device,
                hidden_dropout=0.,
                attention_dropout=0.,
            )
    if hasattr(args, 'model_parallel_size'):
        b['model_parallel_size'] = args.model_parallel_size
    return argparse.Namespace(**deepcopy(b), **vars(minimal_args))

class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        vit_args = override_dist_dtype_device_args(args, args.eva_args)
        self.vit_model = EVA2CLIPModel(EVA2CLIPModel.get_args(**vars(vit_args)))
        self.in_features = 1792
        self.out_features = 1792
        self.conv = nn.Conv2d(in_channels=self.in_features, out_channels=self.out_features, kernel_size=2, stride=2)
        self.linear_proj = GLU(args, self.out_features)
        self.image_length = args.image_length
        self.boi = nn.Parameter(torch.zeros(1, 1, args.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, args.hidden_size))

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        vision_inputs = {}
        for k in kw_args:
            if k.startswith('vision_') and k != 'vision_expert_mask':
                vision_inputs[k[7:]] = kw_args[k]
        if input_ids.shape[1] == 1 or not vision_inputs:
            return self.transformer.word_embeddings(input_ids)
        image_emb = self.vit_model(**vision_inputs)[0]

        b, s, e = image_emb.shape
        grid_size = int(s ** 0.5)
        image_emb = image_emb.view(b, grid_size, grid_size, e).permute(0, 3, 1, 2)
        image_emb = self.conv(image_emb)
        image_emb = image_emb.flatten(2).transpose(1, 2)

        image_emb = self.linear_proj(image_emb)

        image_embed_mask = kw_args['image_embed_mask']
        word_embedding = self.transformer.word_embeddings(input_ids).clone()
        word_embedding[image_embed_mask.bool()] = torch.cat([self.boi.repeat(len(image_emb), 1, 1), image_emb, self.eoi.repeat(len(image_emb), 1, 1)], dim=1).reshape(-1, image_emb.shape[-1])
        return word_embedding.contiguous()


class CogVLM2Model(LLaMAModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))
        self.del_mixin("mlp")
        self.add_mixin("mlp", LlamaVisionExpertFCMixin(args.hidden_size, args.inner_hidden_size, args.num_layers, 32))
        self.del_mixin("rotary")
        self.add_mixin("rotary", LlamaVisionExpertAttnMixin(args.hidden_size, args.num_attention_heads, args.num_layers, 32))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CogVLM', 'CogVLM Configurations')
        group.add_argument('--image_length', type=int, default=256)
        group.add_argument('--eva_args', type=json.loads, default={})
        return super().add_model_specific_args(parser)

    def forward(self, input_ids, vision_expert_mask, image_embed_mask, **kwargs):
        if input_ids.shape[1] > 1:
            return super().forward(input_ids=input_ids, vision_expert_mask=vision_expert_mask, image_embed_mask=image_embed_mask, **kwargs)
        return super().forward(input_ids=input_ids, **kwargs)


class FineTuneTrainCogVLMModel(CogVLMModel):
    def __init__(self, args, transformer=None, **kw_args):
        super().__init__(args, transformer=transformer, **kw_args)
        self.args = args
        # If you want to use model parallel with a mp_size=1 checkpoint, and meanwhile you also want to use lora,
        # you have to add_mixin after loading model checkpoint.
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CogVLM-finetune', 'CogVLM finetune Configurations')
        group.add_argument('--pre_seq_len', type=int, default=8)
        group.add_argument('--lora_rank', type=int, default=10)
        group.add_argument('--use_ptuning', action="store_true")
        group.add_argument('--use_lora', action="store_true")
        group.add_argument('--use_qlora', action="store_true")
        group.add_argument('--layer_range', nargs='+', type=int, default=None)
        return super().add_model_specific_args(parser)


from sat.model.finetune import PTuningV2Mixin
from sat.model.finetune.lora2 import LoraMixin
class FineTuneTestCogVLM2Model(CogVLM2Model):
    def __init__(self, args, transformer=None, **kw_args):
        super().__init__(args, transformer=transformer, **kw_args)
        if args.use_ptuning:
            self.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
        if args.use_lora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
            self.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range), reinit=True)
        elif args.use_qlora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        self.args = args
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CogVLM-finetune', 'CogVLM finetune Configurations')
        group.add_argument('--pre_seq_len', type=int, default=8)
        group.add_argument('--lora_rank', type=int, default=10)
        group.add_argument('--use_ptuning', action="store_true")
        group.add_argument('--use_lora', action="store_true")
        group.add_argument('--use_qlora', action="store_true")
        group.add_argument('--layer_range', nargs='+', type=int, default=None)
        return super().add_model_specific_args(parser)
