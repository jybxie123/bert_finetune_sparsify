import torch
import evaluate

metrics = evaluate.load("accuracy") 
# metrics.reset()
print('metrics',type(metrics))