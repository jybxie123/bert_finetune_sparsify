# training dataset
import os
from src.config.training_config import train_config as TRAIN_CONFIG
import src.self_def_datasets.custom_dataset as custom_dataset
# add transformer into path
from transformers import AutoConfig
from src.models.bert.modeling_bert import BertForSequenceClassification
from transformers import AutoConfig, AutoModelForSequenceClassification
import transformers
import evaluate
import torch
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import fire
from src.utils.train_utils import train
from src.utils.config_utils import update_config

def main(**kwargs):
    # config and fsdp_config
    train_config = TRAIN_CONFIG()
    update_config((train_config), **kwargs)

    train_data = torch.load(os.path.join(train_config.dataset_path, f'train_data_{train_config.dataset_length}.pt'))
    eval_data = torch.load(os.path.join(train_config.dataset_path, f'eval_data_{train_config.dataset_length}.pt'))
    test_data = torch.load(os.path.join(train_config.dataset_path, f'test_data_{train_config.dataset_length}.pt'))

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        num_workers=train_config.num_workers_dataloader,
        batch_size=train_config.batch_size_training,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_data,
            num_workers=train_config.num_workers_dataloader,
            # pin_memory=True,
            batch_size=train_config.batch_size_val,
            # **val_dl_kwargs,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            num_workers=train_config.num_workers_dataloader,
            # pin_memory=True,
            batch_size=train_config.batch_size_val,
            # **val_dl_kwargs,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if train_config.model_type == 'finetuned':
        config = AutoConfig.from_pretrained(f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased")
        model = BertForSequenceClassification.from_pretrained( # BertForSequenceClassification
            f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased",
            sparse_mode = train_config.mode,
            keep_frac = train_config.keep_frac,
            is_sparse_softmax = train_config.is_sparse_softmax,
            is_sparse_layer_norm = train_config.is_sparse_layer_norm,
            num_labels=5,
            config=config)
        ckpt = torch.load(f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased/bert-base-cased-{train_config.ckpt_idx}.pt")
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        origin_model = transformers.AutoModelForSequenceClassification.from_pretrained( # BertForSequenceClassification
            train_config.model_name,
            num_labels=5)
        model = BertForSequenceClassification.from_pretrained(
            train_config.model_name, 
            sparse_mode = train_config.mode, 
            keep_frac = train_config.keep_frac, 
            is_sparse_softmax = train_config.is_sparse_softmax, 
            is_sparse_layer_norm = train_config.is_sparse_layer_norm, 
            num_labels=5)
        model.load_state_dict(origin_model.state_dict(), strict=False)
        del origin_model
    model.to(device)
    # metric = evaluate.load("accuracy")
    print(next(model.parameters()).device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config
    )
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)


# trainer self write
    