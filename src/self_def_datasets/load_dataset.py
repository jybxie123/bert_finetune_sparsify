from datasets import load_dataset
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune_sparsify/src')
from config.training_config import train_config as TRAIN_CONFIG
import torch
import os
from self_def_datasets.custom_dataset import SelfDefDataset

dataset = load_dataset("yelp_review_full")
train_config = TRAIN_CONFIG()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 这个样例数据集不大，300M
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(20000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))
train_data = SelfDefDataset(small_train_dataset)
eval_data = SelfDefDataset(small_eval_dataset)
print(3)
torch.save(train_data, os.path.join(train_config.dataset_path, 'train_data.pt'))
torch.save(eval_data, os.path.join(train_config.dataset_path, 'eval_data.pt'))
