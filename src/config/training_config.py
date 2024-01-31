# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_type: str= 'pretrained' # pretrained' # gelu to relu
    hidden_act: str = 'gelu'
    model_name: str="bert-base-cased"
    mode = 'normal'
    run_validation: bool=True
    batch_size_training: int=64
    val_batch_size: int=64
    batching_strategy: str="packing" #alternative: padding
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=5
    num_workers_dataloader: int=4
    lr: float=5e-5
    weight_decay: float=0.01
    gamma: float= 0.85
    seed: int=42
    mixed_precision: bool=True
    dataset_path = "/disk3/Haonan/yanbo_random/bert_finetune_sparsify/src/self_def_datasets"
    dataset_name = "yelp_review_full"
    output_dir: str = "/disk3/Haonan/yanbo_random/bert_finetune_sparsify/metrics"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    ckpt_path: str="/disk3/Haonan/yanbo_random/bert_finetune_sparsify/checkpoint/train_yelp_gelu_100000_sparsify_0" # will be used if using FSDP
    load_ckpt_path = "/disk3/Haonan/yanbo_random/bert_finetune_sparsify/checkpoint/train_basic_yelp_gelu_100000_sparsity_0" # gelu
    log_path = "/disk3/Haonan/yanbo_random/bert_finetune_sparsify/logs/train_basic_yelp"
    save_optimizer: bool=False 
    save_metrics: bool = False # saves training metrics to a json file for later plotting

    # checkpoint
    ckpt_idx: int = 6

    enable_fsdp: bool=True # False 我得用多gpu，不然跑的巨慢。
    use_fp16: bool=True # False
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    low_cpu_fsdp: bool=False # 这个尽量别开，会慢很多
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    use_pefted_model: str = None


from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT # SHARDED_STATE_DICT  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool=False # True
    fsdp_cpu_offload: bool=False
    pure_bf16: bool = False # 太慢了，尝试提速
    optimizer: str= "AdamW"
    
