import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from dataset import ConversationDataset

def cleanup():
    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser()

    # learning parameters
    parser.add_argument("--lr",
                        type=float,
                        default=2e-4, help="Learning rate for the optimizer"
                        )
    parser.add_argument("--num_epochs",
                        type=int,
                        default=50,
                        help="Number of epochs to train for"
                        )

    # dataset parameters
    parser.add_argument("--dataset_dir",
                        type=str,
                        help="Directory for the dataset")
    parser.add_argument("--train_rate",
                        type=float,
                        default=0.8,
                        help="Training split ratio")
    parser.add_argument("--save_freq",
                        type=int,
                        default=5,
                        help="Frequency of saving the model after epochs")
    parser.add_argument("--save_dir",
                        type=str,
                        default="output",
                        help="Directory to save the training outputs")

    # model parameters
    parser.add_argument("--model_dir",
                        type=str,
                        help="Directory where the model is stored")
    parser.add_argument("--tokenizer_dir",
                        type=str,
                        help="Directory for the tokenizer")
    parser.add_argument("--max_length",
                        type=int,
                        default=512,
                        help="Frequency of saving the model after epochs")

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank for LoRA layers")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha value for LoRA")

    # DDP specific parameters
    parser.add_argument("--per_device_train_batch_size",
                        type=int,
                        default=4,
                        help="Input batch size for training per device")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=4,
                        help="Number of operations to accumulate before performing a backward/update pass.")

    return parser.parse_args()


def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, local_rank, args):
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}", position=0, leave=True)):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            print("loss:", total_loss)

        dist.barrier()

        if local_rank == 0:
            print(f"Epoch {epoch} loss: {total_loss / len(train_dataloader)}")

        # Evaluation Step
        model.eval()
        total_eval_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True)):
            with torch.no_grad():
                outputs = model(**batch)
            total_eval_loss += outputs.loss.item()

        total_eval_loss /= len(eval_dataloader)
        dist.reduce(torch.tensor(total_eval_loss).cuda(), dst=0)

        if local_rank == 0:
            print(f"Avg evaluation loss: {total_eval_loss}")

        if local_rank == 0 and (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            model.module.save_pretrained(save_path)
            print(f"Model saved to {save_path}")


def finetune(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend='nccl',
        rank=local_rank,
        world_size=torch.cuda.device_count()
    )
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(f'cuda:{local_rank}')

    dataset = ConversationDataset(root_dir=args.dataset_dir,
                                  tokenizer=tokenizer,
                                  model=model,
                                  config=config,
                                  max_length=args.max_length,
                                  device=f'cuda:{local_rank}')

    # loading dataset
    train_size = int(args.train_rate * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=dataset.custom_collate_fn,
        drop_last=True
    )

    eval_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
    )

    eval_dataloader = DataLoader(
        val_dataset,
        sampler=eval_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=dataset.custom_collate_fn
    )

    peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                             inference_mode=False,
                             r=args.lora_rank,
                             target_modules=[
                                 "vision_expert_query_key_value",
                                 "q_proj", "k_proj", "v_proj",
                                 "query_key_value",
                                 "language_expert_query_key_value"
                             ],
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs)
    )
    model = DDP(model, find_unused_parameters=True)
    train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, local_rank, args)

    if local_rank == 0:
        save_path = os.path.join(args.save_dir, "final_model")
        model.module.save_pretrained(save_path)

    cleanup()


if __name__ == "__main__":
    args = get_args()
    finetune(args)
