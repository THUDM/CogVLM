import os
import torch
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from collections import defaultdict
from functools import partial

from utils.models import FineTuneTestCogAgentModel
from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor


def data_collator(examples, cross_image_processor=None):
    def to_tensor(value):
        """Converts lists or numpy arrays to tensors."""
        if isinstance(value, list):
            return torch.tensor(value)
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value
    
    def concatenate_tensors(attribute, key):
        """Concatenates tensors for a specific attribute and key."""
        if attribute is None:
            return torch.cat([ex[key] for ex in examples if isinstance(ex[key], torch.Tensor)])
        else:
            return torch.cat([ex[attribute][key] for ex in examples if isinstance(ex[attribute][key], torch.Tensor)])

    # Convert all lists and numpy arrays in examples to tensors
    for example in examples:
        for key, value in example.items():
            example[key] = to_tensor(value)

    # Extract and concatenate attributes from examples
    img_args = {}
    for attribute in ['vision', 'cross']:
        if attribute == 'cross' and cross_image_processor is None:
            continue

        if attribute in examples[-1]:  # Using the last example as reference
            for key in examples[-1][attribute]:
                tensor_key = f"{attribute}_{key}"
                tensors_to_concatenate = [ex[attribute][key] for ex in examples if isinstance(ex[attribute][key], torch.Tensor)]
                if tensors_to_concatenate:
                    img_args[tensor_key] = concatenate_tensors(attribute, key)
                else:
                    img_args[tensor_key] = examples[-1][attribute][key]

    # Remove 'vision' and 'cross' keys from examples
    for example in examples:
        example.pop('vision', None)
        example.pop('cross', None)

    # Create model_args by concatenating tensors and copying other attributes
    model_args = {key: concatenate_tensors(None, key) 
                  if isinstance(examples[-1][key], torch.Tensor) else examples[-1][key] 
                  for key in examples[-1]
                  }
    
    # Merge img_args into model_args
    model_args.update(img_args)
    return model_args

def broadcast_auto(data_dict):
    # Classify keys based on their data type
    tensor_keys_by_dtype = defaultdict(list)
    non_tensor_keys = []

    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_keys_by_dtype[value.dtype].append(key)
        else:
            non_tensor_keys.append(key)

    # Broadcast tensor data and collect in a new dictionary
    broadcasted_data = {}
    for dtype, keys in tensor_keys_by_dtype.items():
        broadcasted_data.update(mpu.broadcast_data(keys, data_dict, dtype))

    # Add non-tensor data to the new dictionary
    for key in non_tensor_keys:
        broadcasted_data[key] = data_dict[key]

    return broadcasted_data

def get_batch(data_iterator, args, timers):
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = broadcast_auto(data)
    for k in data_b:
        if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
            if args.fp16:
                data_b[k] = data_b[k].half()
            elif args.bf16:
                data_b[k] = data_b[k].bfloat16()
    return data_b

from torch.nn import CrossEntropyLoss
import numpy as np

from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy


def chat(model, tokenizer, tokens,
         max_length: int = 1800, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
    inputs = tokens.to(model.parameters().__next__().device)[0]
    seq = torch.cat(
        [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
    # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
    #                               num_beams=num_beams, consider_end=True)
    get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy,
        get_masks_and_position_ids=get_func,
        **kwargs
    )[0]  # drop memory

    return output


def forward_step_eval(data_iterator, model, args, timers):
    def compute_metrics(eval_preds):
        preds, labels, device = eval_preds
        preds = preds.unsqueeze(0)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "acc": [],
            "acc_w/o_case": [],
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            if args.rank == 0:
                print('pred', pred, 'label', label, flush=True)
            if pred == label:
                score_dict['acc'].append(1.)
            else:
                score_dict['acc'].append(0.)
            if pred.lower() == label.lower():
                score_dict['acc_w/o_case'].append(1.)
            else:
                score_dict['acc_w/o_case'].append(0.)
            

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    context_len = int(data_b['context_length'][0])
    tokens = data_b['input_ids'][:, :context_len]
    data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
    data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
    data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]

    data_b.pop('input_ids')
    data_b.pop('attention_mask')
    data_b.pop('position_ids')
    labels = data_b.pop('labels')
    qid = data_b.pop('question_id')

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    outputs = chat(model, tokenizer, tokens, **data_b)[0][context_len:]
    # print(outputs)
    model.del_mixin('auto-regressive')

    return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in
                                                    compute_metrics(
                                                        (outputs.cpu(), labels.cpu(), outputs.device)).items()}


from torch.nn import CrossEntropyLoss
def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    labels = data_b.pop('labels')
    timers('batch generator').stop()
    logits = model(**data_b)[0]
    lm_logits = logits.to(torch.float32)
    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.to(torch.float32)

    return loss, {'loss': loss}

from utils.utils import ItemDataset
def create_dataset_function(image_processor, text_processor, cross_image_processor, path, args):
    dataset = ItemDataset(image_processor, text_processor, args, path, cross_image_processor=cross_image_processor)
    return dataset

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
    py_parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    py_parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    py_parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser = FineTuneTestCogAgentModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    if args.use_qlora:
        args.device = 'cpu'

    model, args = FineTuneTestCogAgentModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    if args.use_qlora and torch.cuda.is_available():
        model = model.to('cuda')
    from utils.utils import llama2_tokenizer
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(args.cross_image_pix)
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=partial(create_dataset_function, image_processor, text_processor, cross_image_processor), collate_fn=partial(data_collator, cross_image_processor=cross_image_processor), forward_step_eval=forward_step_eval)