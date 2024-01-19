# training dataset
from datasets import load_dataset
from datasets import load_dataset, load_metric, list_metrics

import sys 
# add transformer into path
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune/transformers/src')
import transformers
print(transformers.__file__)


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

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from pathlib import Path
from utils.train_utils import train, evaluation
from training_config import train_config as TRAIN_CONFIG

# load_ckpt_path = "/disk3/Haonan/yanbo_random/bert_finetune/checkpoint/train_basic_yelp" # gelu
ckpt_path = "/disk3/Haonan/yanbo_random/bert_finetune/checkpoint/train_basic_yelp_original_trainer"

log_path = "/disk3/Haonan/yanbo_random/bert_finetune/logs/train_basic_yelp"
print('----> load dataset')
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 这个样例数据集不大，300M

# import torch
# torch.cuda.is_available()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
small_eval_dataset = eval_dataset.select(range(10000))
# small_train_dataset = small_train_dataset.to(device)
# small_eval_dataset = small_eval_dataset.to(device)
# test_dataset = eval_dataset.select(range(20000,30000))
# test_dataset.shape

# use ft model
print('----> load model')
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5) # gelu
# model = AutoModelForSequenceClassification.from_pretrained(f"{load_ckpt_path}/checkpoint-18750", num_labels=5)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
model.to(device)
# print(next(model.parameters()).device)

train_config = TRAIN_CONFIG()

optimizer = optim.AdamW(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=train_config.weight_decay,
)
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

training_args = TrainingArguments(
    output_dir=ckpt_path,          # 输出目录
    num_train_epochs=3,              # 总的训练轮数
    per_device_train_batch_size=16,  # 每个 GPU 或 CPU 的训练批次大小
    per_device_eval_batch_size=64,   # 每个 GPU 或 CPU 的评估批次大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,                # 多少步骤记录一次日志
)

# Start the training process
# results = train(
#     model,
#     small_train_dataset,
#     small_eval_dataset,
#     tokenizer,
#     optimizer,
#     scheduler,
#     train_config.gradient_accumulation_steps,
#     train_config
# )
# [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
# this training method does not support the forward step counter
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
print(trainer.args.device)
trainer.train()
trainer.save_model(ckpt_path)

