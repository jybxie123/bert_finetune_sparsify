# training dataset
from datasets import load_dataset
from datasets import load_dataset, load_metric, list_metrics

import sys 
# add transformer into path
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune/transformers/src')
import transformers
print(transformers.__file__)

from transformers import AutoModelForSequenceClassification
import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
with open('/disk3/Haonan/yanbo_random/bert_finetune/model_structure/bert_structure_replaced.txt', 'w') as file:
    for name, module in model.named_modules():
        file.write(f"Module Name: {name}\n")
        file.write(f"{module}\n")

