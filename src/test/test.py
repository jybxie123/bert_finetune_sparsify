# training dataset
from datasets import load_dataset
from datasets import load_dataset, load_metric, list_metrics
from models.bert.modeling_bert import BertForSequenceClassification
import pbar
import wandb
import sys 
# add transformer into path
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune/transformers/src')
import transformers
print(transformers.__file__)

import socket
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
import numpy as np
import evaluate
import torch
import os
from config.training_config import train_config as TRAIN_CONFIG
# config
train_config = TRAIN_CONFIG()
# dataset
test_data = torch.load(os.path.join(train_config.dataset_path, 'eval_data.pt'))
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    num_workers=train_config.num_workers_dataloader,
    batch_size=train_config.batch_size_test,
)

metrics = evaluate.load("accuracy")
# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# test_dataset.to(device)

wandb.init(
    config={
        'learning_rate': train_config.lr,
        'batch_size': train_config.batch_size_test,
        'num_epochs': train_config.num_epochs,
        'dataset': train_config.dataset_name,
        'model': train_config.model_name,
        'mixed_precision': train_config.mixed_precision,
    },
    project='bert-sparsity-0',
    entity='bert-sparsify',
    notes=socket.gethostname(),
    name=train_config.expr_name,
    job_type="test",
    reinit=True
    )


for epoch_idx in range(train_config.num_epochs):
    config = AutoConfig.from_pretrained(f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased")
    model = BertForSequenceClassification.from_pretrained( # BertForSequenceClassification
        f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased",
        sparse_mode = train_config.mode,
        keep_frac = train_config.keep_frac,
        is_sparse_softmax = train_config.is_sparse_softmax,
        config=config)
    ckpt = torch.load(f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased/bert-base-cased-{train_config.epoch_idx}.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to('cuda:0')
            outputs = model(**batch)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)
            metrics.add_batch(predictions=predicted_labels, references=batch["labels"])
            pbar.update(1)
            pbar.set_description(f"Test: step {step}/{len(test_dataloader)} completed")
        pbar.close()
        wandb.log({"accuracy": metrics.compute()['accuracy']}, step=epoch_idx)

        accu = metrics.compute()
        print(accu)
    wandb.finish()
