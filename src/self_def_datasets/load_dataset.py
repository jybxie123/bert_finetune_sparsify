from datasets import load_dataset
from transformers import AutoTokenizer
import sys
sys.path.append('/home/runtao/DL_Team_Proj/bert_finetune_sparsify/src')
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
small_train_dataset = tokenized_datasets["train"].shuffle(seed=13).select(range(train_config.dataset_length))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=13).select(range(int(train_config.dataset_length/5)))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=13).select(range(int(train_config.dataset_length/5), int(train_config.dataset_length/5)*2))
train_data = SelfDefDataset(small_train_dataset)
eval_data = SelfDefDataset(small_eval_dataset)
test_data = SelfDefDataset(small_test_dataset)
print('load dataset done')
torch.save(train_data, os.path.join(train_config.dataset_path, f'train_data_{train_config.dataset_length}.pt'))
torch.save(eval_data, os.path.join(train_config.dataset_path, f'eval_data_{train_config.dataset_length}.pt'))
torch.save(test_data, os.path.join(train_config.dataset_path, f'test_data_{train_config.dataset_length}.pt'))
