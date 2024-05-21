# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_type: str = "pretrained"  # pretrained' # gelu to relu
    hidden_act: str = "relu"  # 'relu' 'relu_inplace'
    model_name: str = "bert-base-cased"
    mode = "sfrl"  # ['nosp', 'rand', 'norm', 'bkrz', 'vrce', 'nml1', 'nml2', 'sfrl'] sfrl shifted relu
    keep_frac: float = 0.1
    run_validation: bool = True
    batch_size_training: int = 16
    batch_size_val: int = 16
    batch_size_test: int = 16
    batching_strategy: str = "packing"  # alternative: padding
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 10
    num_workers_dataloader: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    gamma: float = 0.85
    seed: int = 42
    mixed_precision: bool = True
    dataset_name = "yelp_review_full"
    dataset_length: int = 20000
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    is_sparse_softmax = False
    is_sparse_layer_norm = False
    root = "/home/bizon/yanbo_random/assi_bert/bert_finetune_sparsify"
    dataset_path: str = (
        f"{root}/src/self_def_datasets"
    )
    expr_name: str = (
        f"custom_{mode}_{dataset_name}_{dataset_length}_{hidden_act}_keep_frac_{keep_frac}_epoch_{num_epochs}__{is_sparse_softmax}_st_{is_sparse_layer_norm}_ln_1"  # _var_no_sm_no_weight' #_norm_no_sm_no_weight_no_double_matmul_add_mem_profiler' # # _norm_no_sm_no_weight
    )
    # expr_name: str = f'bert_yelp_{hidden_act}_{dataset_length}_10_custom_{mode}_{is_sparse_softmax}_st_{is_sparse_layer_norm}_ln_{keep_frac}_keep_frac_inf_norm_epoch_{num_epochs}' #_var_no_sm_no_weight' # _norm_no_sm_no_weight
    ckpt_path: str = (
        f"{root}/checkpoint"  # will be used if using FSDP
    )
    load_ckpt_path = (
        f"{root}/checkpoint"  # gelu
    )
    output_dir: str = (
        f"{root}/metrics"
    )
    log_path = f"{root}/logs"
    save_optimizer: bool = False
    save_metrics: bool = (
        False  # saves training metrics to a json file for later plotting
    )

    # checkpoint
    ckpt_idx: int = 4

    # unused multi gpu training
    enable_fsdp: bool = True  # False
    use_fp16: bool = True  # False
    use_fast_kernels: bool = (
        False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    )
    low_cpu_fsdp: bool = False  # 这个尽量别开，会慢很多
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    use_pefted_model: str = None
