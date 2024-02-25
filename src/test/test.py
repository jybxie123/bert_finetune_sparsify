# training dataset
from datasets import load_dataset
from datasets import load_dataset, load_metric, list_metrics

import sys 
# add transformer into path
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune_sparsify/src')
import transformers
print(transformers.__file__)

from models.bert.modeling_bert import BertForSequenceClassification
from transformers import AutoConfig
import numpy as np
import evaluate
import os
import torch
from config.training_config import train_config as TRAIN_CONFIG
from tqdm import tqdm
import socket
import wandb
# config
train_config = TRAIN_CONFIG()

test_data = torch.load(os.path.join(train_config.dataset_path, 'test_data.pt'))
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    num_workers=train_config.num_workers_dataloader,
    batch_size=train_config.batch_size_test,
)
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

wandb.init(
    config={
        'learning_rate': train_config.lr,
        'batch_size': train_config.batch_size_training,
        'num_epochs': train_config.num_epochs,
        'dataset': train_config.dataset_name,
        'model': train_config.model_name,
        'scheduler': "StepLR",
        'mixed_precision': train_config.mixed_precision,
    },
    project='bert-sparsity-test',
    entity='bert-sparsify',
    notes=socket.gethostname(),
    name=train_config.expr_name,
    job_type="test",
    reinit=True
    )


predictions = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 10 # old version is 5
for ckpt_idx in range(epoch):
    config = AutoConfig.from_pretrained(f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased")
    model = BertForSequenceClassification.from_pretrained( # BertForSequenceClassification
        train_config.model_name, 
        sparse_mode = train_config.mode,
        keep_frac = train_config.keep_frac,
        is_sparse_softmax = train_config.is_sparse_softmax,
        is_sparse_layer_norm = train_config.is_sparse_layer_norm,
        config=config)
    ckpt = torch.load(f"{train_config.load_ckpt_path}/{train_config.expr_name}/bert-base-cased/bert-base-cased-{ckpt_idx}.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    metrics = evaluate.load("accuracy")
    with torch.no_grad():
        step = 0
        test_accu = [] 
        total_length = len(test_dataloader)
        pbar = tqdm(colour="blue", desc=f"Test", total=total_length, dynamic_ncols=True)
        for step, batch in enumerate(test_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to('cuda:0')
            outputs = model(**batch)
            logits = outputs.logits
            predicted_labels = torch.tensor(torch.argmax(logits, dim=1))
            predictions.append(predicted_labels)
            metrics.add_batch(predictions=predicted_labels, references=batch["labels"])
            pbar.update(1)
            pbar.set_description(f"Test, step {step}/{len(test_dataloader)} completed ")
        pbar.close()
        accu = metrics.compute()
        print(accu)
        wandb.log({"test_accuracy": accu['accuracy']}, step=ckpt_idx)
