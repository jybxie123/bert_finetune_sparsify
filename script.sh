cd bert_finetune_sparsify
# 先加载并打包数据集，只需要做一次就行。
python src/self_def_datasets/load_dataset.py

# 微调模型
CUDA_VISIBLE_DEVICES=0 python src/train.py

# 测试模型 见readme。

0 -> nosp
1 -> norm
2 -> nml1

3 -> rand
4 -> bkrz
5 -> nml2


