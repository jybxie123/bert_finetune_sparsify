# training dataset
from datasets import load_dataset, load_metric, list_metrics


import os
import sys

# add transformer into path
sys.path.insert(0, '/home/bizon/yanbo_random/bert_finetune_sparsify_new_version/transformers/src')
import transformers
print(transformers.__file__)

from transformers import  AutoConfig
from transformers.models.bert.configuration_bert import BertConfig 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from pkg_resources import packaging

import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
import fire
import self_def_datasets.dataset as sds
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from transformers.models.bert.modeling_bert import BertLayer
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from functools import partial

from pathlib import Path
from training_config import train_config as TRAIN_CONFIG
from training_config import fsdp_config as FSDP_CONFIG

from utils.train_utils import train, setup, get_policies, freeze_transformer_layers, clear_gpu_cache, setup_environ_flags
from utils.config_utils import update_config
from utils.fsdp_utils import fsdp_auto_wrap_policy



load_ckpt_path = "/home/bizon/yanbo_random/bert_finetune_sparsify_new_version/checkpoint/train_basic_yelp_gelu_100000_sparsity_0" # gelu

log_path = "/home/bizon/yanbo_random/bert_finetune_sparsify_new_version/logs/train_basic_yelp"
def main(**kwargs):
    # config and fsdp_config
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    use_cache = False if train_config.enable_fsdp else None

    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # 这个样例数据集不大，300M
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
    train_data = sds.SelfDefDataset(small_train_dataset)
    eval_data = sds.SelfDefDataset(small_eval_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        num_workers=train_config.num_workers_dataloader,
        batch_size=train_config.batch_size_training,
        # pin_memory=True,
        # **train_dl_kwargs,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_data,
            num_workers=train_config.num_workers_dataloader,
            # pin_memory=True,
            batch_size=train_config.val_batch_size,
            # **val_dl_kwargs,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if train_config.model_type == 'finetuned':
        config = AutoConfig.from_pretrained(f"{load_ckpt_path}/bert-base-cased")
        print('act_layer : ',config.hidden_act)
        model = AutoModelForSequenceClassification.from_config(config=config)
        ckpt = torch.load(f"{load_ckpt_path}/bert-base-cased/bert-base-cased-{train_config.ckpt_idx}.pt")
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # use original model
        model = AutoModelForSequenceClassification.from_pretrained(
            train_config.model_name,
            num_labels=5,
            # load_in_8bit=True if train_config.quantization else None,
            # device_map="auto" if train_config.quantization else None,
            # use_cache=use_cache,
        )
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, BertLayer)
        # add model to fsdp
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=device, recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            print("-----> Applying activation checkpointing to model")
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(submodule, BertLayer)
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to(device)

    # mixed precision
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # metric=load_metric("accuracy")
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    print(next(model.parameters()).device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    # optimizer = optim.SparseAdam(
    #     model.parameters(),
    #     lr=train_config.lr,
    # )

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)


# trainer self write
    