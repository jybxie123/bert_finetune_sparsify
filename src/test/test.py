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
from transformers import AutoConfig
import numpy as np
import evaluate
import torch
from config.training_config import train_config as TRAIN_CONFIG
# ckpt_path = "/disk3/Haonan/yanbo_random/bert_finetune/checkpoint/train_basic_yelp"
ckpt_path = "/disk3/Haonan/yanbo_random/bert_finetune/checkpoint/train_basic_yelp_gelu_100000_sparsify_1"
# config
train_config = TRAIN_CONFIG()

# dataset
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
test_dataset = eval_dataset.select(range(3000,4000))
# 这个样例数据集不大，300M
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# test_dataset.to(device)
# test
# model_list = [6250,12500,18750,25000,31250]
# model_list = [63,126,189,252,315]
# model_list = [6250,12500,18750,25000,31250]

# for model_idx in model_list:
if train_config.model_type == 'finetuned':
    config = AutoConfig.from_pretrained(f"{ckpt_path}/bert-base-cased")
    model = AutoModelForSequenceClassification.from_config(config=config)
    ckpt = torch.load(f"{ckpt_path}/bert-base-cased/bert-base-cased-{train_config.ckpt_idx}.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    # use fine tuned model
    # model = AutoModelForSequenceClassification.from_pretrained(f"{ckpt_path}/bert-base-cased", )
else:
    # use original model
    model = AutoModelForSequenceClassification.from_pretrained(
        train_config.model_name,
        num_labels=5,
        # load_in_8bit=True if train_config.quantization else None,
        # device_map="auto" if train_config.quantization else None,
        # use_cache=use_cache,
    )
# model = AutoModelForSequenceClassification.from_pretrained(f"{ckpt_path}/bert-base-cased",num_labels=5)
model.to(device)
model.eval()
predictions = []
with torch.no_grad():
    step = 0
    for batch in test_dataset:
        if step %100 == 0:
            print(step)
        step +=1
        # add batch size?
        inputs = {k: torch.tensor(v,device=device) for k, v in batch.items() if k in tokenizer.model_input_names}
        # print(inputs)
        for k in inputs.keys():
            inputs[k]= inputs[k].unsqueeze(0)
        outputs = model(**inputs)
        # print(outputs)
        prediction = torch.tensor(torch.argmax(outputs.logits, dim=-1))
        predictions.append(prediction)
    metric.add_batch(predictions=predictions, references=test_dataset["label"])
score = metric.compute() 
print(score)

# best epoch 3.
