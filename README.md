# Bert Finetune Backward Sparsify
*This is our source code of Bert finetune with backward sparsify in column form way.*
## Introduction
Our code is for sparsifying the gradient of backpropagation. To achieve this goal and get a better performance than the former works, we find a new method of sparsify based in column selection and evaluation of feature of input.
![image](https://github.com/jybxie123/bert_finetune_sparsify/assets/66007115/f1eceb16-632a-44bd-ad4b-334cb803a838)


## Framework
This repo uses self defined dataloader, trainer, and evaluation framework.
We rewrite the transformers bert model into our type.

## Requirement
Go to the main folder, find the requirements.txt and run the following:
```
pip install -r requirements.txt
```
If it doesn't work, give me an issue.
## Training
Before you run your code, edit your training_config to make sure it runs as you think.
A good example:
```
# load your dataset into your dataset folder
python src/self_def_dataset/load_dataset.py
# finetune
nvidia-smi # find your available gpu index
CUDA_VISIBLE_DEVICES=<YOUR GPU INDEX, SPLIT BY ','> python  <YOUR REPO PATH>/src/train.py
```
If you need to edit your config, please check the file : <YOUR REPO PATH>/src/config/training_config.py 


##  Testing
Testing is quite simple, just go to the main folder, and run the following script:
```
cd src
# if you need to specialise your gpu, use this method: CUDA_VISIBLE_DEVICES=<YOUR GPU INDEX>
python test.py
# test.py is outdated, repairing
```



