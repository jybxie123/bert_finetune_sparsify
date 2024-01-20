# Readme of Bert Finetune Backward Sparsify
*This is our source code of Bert finetune with backward sparsify in column form way.*
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
CUDA_VISIBLE_DEVICES=<YOUR GPU INDEX, SPLIT BY ','> torchrun --nnodes 1 --nproc_per_node 2 <YOUR REPO PATH>/src/train.py
```

##  Testing
Testing is quite simple, just go to the main folder, and run the following script:
```
cd src
# if you need to specialise your gpu, use this method: CUDA_VISIBLE_DEVICES=<YOUR GPU INDEX>
python test.py
```



