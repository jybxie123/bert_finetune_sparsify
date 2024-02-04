git clone https://github.com/huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b --model_size 7B --output_dir /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf

# 单gpu
#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name /home/bizon/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --output_dir /home/bizon/yanbo_random/llama-recipes/checkpoint/llama-2-7b-peft-lora/ --save_metrics

# 问题1，用什么来微调，lora好像不行吧？
# 不过感觉又好像可以啊
###实验步骤：先微调结束，看效果 
###然后再relu测试，看看推理是否大幅变差
###再微调一下看效果，如果relu加完能够回去的话，就成功了。

# 单gpu跑不了，死心吧
# 运行成功
CUDA_VISIBLE_DEVICES=0,1,7 torchrun --nnodes 1 --nproc_per_node 3  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --fsdp_config.pure_bf16 --output_dir /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-peft-lora/ --save_metrics
# 画图
python examples/plot_metrics.py --file_path /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-peft-lora/metrics_data_2-2023-12-29_05-16-17.json

# 推理：
python examples/inference.py --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --peft_model /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-peft-lora/ --prompt_file /disk3/Haonan/yanbo_random/llama-recipes/examples/samsum_prompt.txt
# 推理没有数值结果返回，它是使用性的测试。
# 自带的test是单元测试等无关内容。
# 仿照原文进行训练：
#RefinedWeb数据集训练 这里我没做，我用了代码自带的数据集来微调。
#小样本任务进行测试
#这些任务：Arc-E Arc-C Hellaswag BoolQ PIQA LAMBADA TriviaQA WinoGrande SciQ
# lm-eval库很nb，这些task可以一把测掉。

# 这里调的是内置的模型，所以改不了，我需要调本地的model
lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf,parallelize=True,load_in_4bit=True,peft=llama-2-7b-peft-lora --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.4292|±  |0.0145|
|             |       |none  |     0|acc_norm|0.4548|±  |0.0146|
|arc_easy     |Yaml   |none  |     0|acc     |0.7597|±  |0.0088|
|             |       |none  |     0|acc_norm|0.7399|±  |0.0090|
|boolq        |Yaml   |none  |     0|acc     |0.7988|±  |0.0070|
|hellaswag    |Yaml   |none  |     0|acc     |0.5636|±  |0.0049|
|             |       |none  |     0|acc_norm|0.7570|±  |0.0043|
|piqa         |Yaml   |none  |     0|acc     |0.7818|±  |0.0096|
|             |       |none  |     0|acc_norm|0.7867|±  |0.0096|
|winogrande   |Yaml   |none  |     0|acc     |0.7024|±  |0.0128|

# 这里调本地模型成功了。需要检查是不是根据config取的模型
# batch有点太小了，而且我也不想转精度，16bit试试？好像没问题
CUDA_VISIBLE_DEVICES=1,6,7 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,parallelize=True,peft=llama-2-7b-peft-lora --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16


# 对使用relu的重新ft
CUDA_VISIBLE_DEVICES=1,6,7 torchrun --nnodes 1 --nproc_per_node 3  examples/finetuning_tuned_models.py --enable_fsdp --use_peft --peft_method lora --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --use_pefted_model /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-peft-lora/ --fsdp_config.pure_bf16 --output_dir /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-relu-peft-lora/ --save_metrics


改用relu之后确实明显降低了：
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2253|±  |0.0122|
|             |       |none  |     0|acc_norm|0.2747|±  |0.0130|
|arc_easy     |Yaml   |none  |     0|acc     |0.2677|±  |0.0091|
|             |       |none  |     0|acc_norm|0.2795|±  |0.0092|
|boolq        |Yaml   |none  |     0|acc     |0.5847|±  |0.0086|
|hellaswag    |Yaml   |none  |     0|acc     |0.2560|±  |0.0044|
|             |       |none  |     0|acc_norm|0.2576|±  |0.0044|
|piqa         |Yaml   |none  |     0|acc     |0.5424|±  |0.0116|
|             |       |none  |     0|acc_norm|0.5098|±  |0.0117|
|winogrande   |Yaml   |none  |     0|acc     |0.4972|±  |0.0141|

# 对新finetune的模型可视化：
python examples/plot_metrics.py --file_path /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-relu-peft-lora/metrics_data_2-2023-12-30_04-25-04.json

CUDA_VISIBLE_DEVICES=1,6,7 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,parallelize=True,peft=llama-2-7b-relu-peft-lora --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2423|±  |0.0125|
|             |       |none  |     0|acc_norm|0.2918|±  |0.0133|
|arc_easy     |Yaml   |none  |     0|acc     |0.2778|±  |0.0092|
|             |       |none  |     0|acc_norm|0.2917|±  |0.0093|
|boolq        |Yaml   |none  |     0|acc     |0.5994|±  |0.0086|
|hellaswag    |Yaml   |none  |     0|acc     |0.2579|±  |0.0044|
|             |       |none  |     0|acc_norm|0.2593|±  |0.0044|
|piqa         |Yaml   |none  |     0|acc     |0.5441|±  |0.0116|
|             |       |none  |     0|acc_norm|0.5229|±  |0.0117|
|winogrande   |Yaml   |none  |     0|acc     |0.4949|±  |0.0141|
这没恢复啊。。。
我猜测是训练还没到位
这里把epoch改到了10
CUDA_VISIBLE_DEVICES=6,7 torchrun --nnodes 1 --nproc_per_node 2  examples/finetuning_tuned_models.py --enable_fsdp --use_peft --peft_method lora --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --use_pefted_model /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-relu-peft-lora/ --fsdp_config.pure_bf16 --output_dir /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-relu-peft-lora-epoch10/ --save_metrics
#测试relu ft后的结果，仍然没有提升回来。
CUDA_VISIBLE_DEVICES=1,6,7 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,parallelize=True,peft=llama-2-7b-relu-peft-lora-epoch10 --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16

测试直接用原始模型改relu后微调：
用下面的：
CUDA_VISIBLE_DEVICES=1,6,7 torchrun --nnodes 1 --nproc_per_node 3  examples/finetuning_tuned_models.py --enable_fsdp --use_peft --peft_method lora --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --use_pefted_model /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-peft-lora/ --fsdp_config.pure_bf16 --output_dir /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-relu-peft-lora/ --save_metrics
CUDA_VISIBLE_DEVICES=1,6,7 torchrun --nnodes 1 --nproc_per_node 3  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --fsdp_config.pure_bf16 --output_dir /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-only-relu-peft-lora/ --save_metrics

训完直接进行一个测试：
CUDA_VISIBLE_DEVICES=1,6,7 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,parallelize=True,peft=llama-2-7b-only-relu-peft-lora --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2184|±  |0.0121|
|             |       |none  |     0|acc_norm|0.2500|±  |0.0127|
|arc_easy     |Yaml   |none  |     0|acc     |0.4722|±  |0.0102|
|             |       |none  |     0|acc_norm|0.4263|±  |0.0101|
|boolq        |Yaml   |none  |     0|acc     |0.6254|±  |0.0085|
|hellaswag    |Yaml   |none  |     0|acc     |0.3714|±  |0.0048|
|             |       |none  |     0|acc_norm|0.4635|±  |0.0050|
|piqa         |Yaml   |none  |     0|acc     |0.6659|±  |0.0110|
|             |       |none  |     0|acc_norm|0.6812|±  |0.0109|
|winogrande   |Yaml   |none  |     0|acc     |0.5541|±  |0.0140|

最后做一下可视化吧：
python examples/plot_metrics.py --file_path /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-only-relu-peft-lora-epoch10/metrics_data_2-2023-12-30_11-35-34.json

==========================
和师兄讨论后，认为应该是不能用lora


全参数训练
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6, torchrun --nnodes 1 --nproc_per_node 6  examples/finetuning.py --enable_fsdp --model_name checkpoint/llama-2-7b-hf --dist_checkpoint_root_folder checkpoint --dist_checkpoint_folder llama-2-7b-silu-full-param --save_metrics --output_dir /disk3/Haonan/yanbo_random/llama-recipes/matrics/llama-2-7b-silu-full-param
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nnodes 1 --nproc_per_node 3  examples/finetuning.py --enable_fsdp --model_name checkpoint/llama-2-7b-hf --dist_checkpoint_root_folder checkpoint --dist_checkpoint_folder llama-2-7b-relu-full-param --save_metrics --output_dir /disk3/Haonan/yanbo_random/llama-recipes/matrics/llama-2-7b-relu-full-param



# torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /path_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --use_fast_kernels

全参数测试：
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,parallelize=True --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
原预训练模型，silu：
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.4309|±  |0.0145|
|             |       |none  |     0|acc_norm|0.4608|±  |0.0146|
|arc_easy     |Yaml   |none  |     0|acc     |0.7622|±  |0.0087|
|             |       |none  |     0|acc_norm|0.7458|±  |0.0089|
|boolq        |Yaml   |none  |     0|acc     |0.7780|±  |0.0073|
|hellaswag    |Yaml   |none  |     0|acc     |0.5717|±  |0.0049|
|             |       |none  |     0|acc_norm|0.7600|±  |0.0043|
|piqa         |Yaml   |none  |     0|acc     |0.7807|±  |0.0097|
|             |       |none  |     0|acc_norm|0.7894|±  |0.0095|
|winogrande   |Yaml   |none  |     0|acc     |0.6922|±  |0.0130|



更新版本：
Please upgrade to transformers>=4.36 and torch>=2.1.1 

测试代码：
CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf --model_args parallelize=True --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
model=llama-2-7b-silu-full-param-llama-2-7b-hf

accelerate launch -m 

StateDictType.SHARDED_STATE_DICT

python ../../lm-evaluation-harness/lm_eval/__main__.py
修改后的测试代码：
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 accelerate launch -m lm_eval --model hf --model_args pretrained=llama-2-7b-hf,parallelize=True,fsdp=llama-2-7b-silu-full-param-llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16

# 当前finetune的bz
# 重新梳理一遍实验过程：
1. silu形态微调
2. relu形态微调

finetune形态的batchsize：
我采用的是默认的配置
batch_size_training: int=4
val_batch_size: int=1

另一个是llama里
你随便（random pick）找几个layer的transformer的
attention block
看一下那个projection matrix的size 
是那个linear里W的size

# 查看模型结构
python print_model.py --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/


relu的测试代码：目前必须只用一个机器
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=llama-2-7b-relu-full-param-checkpoint/llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
这是silu
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=llama-2-7b-silu-full-param-llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16

|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.4326|±  |0.0145|
|             |       |none  |     0|acc_norm|0.4480|±  |0.0145|
|arc_easy     |Yaml   |none  |     0|acc     |0.7614|±  |0.0087|
|             |       |none  |     0|acc_norm|0.7445|±  |0.0089|
|boolq        |Yaml   |none  |     0|acc     |0.7761|±  |0.0073|
|hellaswag    |Yaml   |none  |     0|acc     |0.5688|±  |0.0049|
|             |       |none  |     0|acc_norm|0.7550|±  |0.0043|
|piqa         |Yaml   |none  |     0|acc     |0.7840|±  |0.0096|
|             |       |none  |     0|acc_norm|0.7922|±  |0.0095|
|winogrande   |Yaml   |none  |     0|acc     |0.6977|±  |0.0129|


hidden layer的稀疏性。算一个起点

recover之后的模型的稀疏性。算一个终点。

多算几个层、几个样本的稀疏程度。finetune的稀疏程度。得多少有点了解。

猜测是会挺稀疏的

把backward算符改掉，很省memory

测vit模型的performance。可能更重要点，根据经验。

vit是一个几种架构，训练方法有很多种。
用clip，一侧是vit 一侧是text，
一个是bert base，另一个是vit/resnet，用vit吧。
用vit给他做稀疏了，bert给冻住。

为什么用clip，因为他本身提出了zero shot的概念。
one-shot这是给一个demo image。image 和image承
clip行是因为他的text和image在一个space里面。

vit或者clip做image net上面的zero shot performance
也可以下一个zero shot的小版本。机器上面好像有

先测一下cotag101上面微调然后测试。

再在imagenet上面微调vit


1.4：继续测试
首先验证测试了多gpu的函数：
1,3,4,7
accelerate launch -m ？
CUDA_VISIBLE_DEVICES=3,6 torchrun --nnodes 1 --nproc_per_node 2 ../../lm-evaluation-harness/lm_eval/__main__.py --model hf --model_args pretrained=llama-2-7b-hf,parallelize=True,fsdp=llama-2-7b-relu-full-param-checkpoint/llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
得用torchrun替换lm_eval才能实现local rank等的设置
还是不行！查了各种源码写法，都试过无法解决keyerror的问题。
而且我是在silu relu两种代码实现的。

目前只能单gpu！
CUDA_VISIBLE_DEVICES=6 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=llama-2-7b-relu-full-param-checkpoint/llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16

重新对模型推理：
CUDA_VISIBLE_DEVICES=6  python examples/inference.py --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --fsdp /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-relu-full-param-checkpoint/llama-2-7b-hf/ --prompt_file /disk3/Haonan/yanbo_random/llama-recipes/examples/samsum_prompt.txt
0.661945238

CUDA_VISIBLE_DEVICES=6 python examples/inference.py --model_name /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-hf/ --fsdp /disk3/Haonan/yanbo_random/llama-recipes/checkpoint/llama-2-7b-silu-full-param-llama-2-7b-hf/ --prompt_file /disk3/Haonan/yanbo_random/llama-recipes/examples/samsum_prompt.txt
0.6965

32个layers全部计算：
silu:0.7287
relu:0.7419

用小样本任务
CUDA_VISIBLE_DEVICES=6 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=llama-2-7b-relu-full-param-checkpoint/llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
什么情况，怎么变差了
0.7019（relu层稀疏度） 
读取的函数是loadpretrained
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2253|±  |0.0122|
|             |       |none  |     0|acc_norm|0.2594|±  |0.0128|
|arc_easy     |Yaml   |none  |     0|acc     |0.4949|±  |0.0103|
|             |       |none  |     0|acc_norm|0.4503|±  |0.0102|
|boolq        |Yaml   |none  |     0|acc     |0.6254|±  |0.0085|
|hellaswag    |Yaml   |none  |     0|acc     |0.3804|±  |0.0048|
|             |       |none  |     0|acc_norm|0.4829|±  |0.0050|
|piqa         |Yaml   |none  |     0|acc     |0.6785|±  |0.0109|
|             |       |none  |     0|acc_norm|0.6888|±  |0.0108|
|winogrande   |Yaml   |none  |     0|acc     |0.5604|±  |0.0139|
silu后改为relu(silu大概率确实稀疏)
0.6370790004730225
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2346|±  |0.0124|
|             |       |none  |     0|acc_norm|0.2747|±  |0.0130|
|arc_easy     |Yaml   |none  |     0|acc     |0.2698|±  |0.0091|
|             |       |none  |     0|acc_norm|0.2858|±  |0.0093|
|boolq        |Yaml   |none  |     0|acc     |0.5988|±  |0.0086|
|hellaswag    |Yaml   |none  |     0|acc     |0.2564|±  |0.0044|
|             |       |none  |     0|acc_norm|0.2606|±  |0.0044|
|piqa         |Yaml   |none  |     0|acc     |0.5446|±  |0.0116|
|             |       |none  |     0|acc_norm|0.5272|±  |0.0116|
|winogrande   |Yaml   |none  |     0|acc     |0.4917|±  |0.0141|

原silu的预测。
0.6961
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.4317|±  |0.0145|
|             |       |none  |     0|acc_norm|0.4497|±  |0.0145|
|arc_easy     |Yaml   |none  |     0|acc     |0.7664|±  |0.0087|
|             |       |none  |     0|acc_norm|0.7437|±  |0.0090|
|boolq        |Yaml   |none  |     0|acc     |0.8128|±  |0.0068|
|hellaswag    |Yaml   |none  |     0|acc     |0.5573|±  |0.0050|
|             |       |none  |     0|acc_norm|0.7581|±  |0.0043|
|piqa         |Yaml   |none  |     0|acc     |0.7851|±  |0.0096|
|             |       |none  |     0|acc_norm|0.7943|±  |0.0094|
|winogrande   |Yaml   |none  |     0|acc     |0.7009|±  |0.0129|

我不服气，重新再跑一下relu的情况，还是一样的

之前的实验：relu的参数silu的激活：
0.6957
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.4352|±  |0.0145|
|             |       |none  |     0|acc_norm|0.4497|±  |0.0145|
|arc_easy     |Yaml   |none  |     0|acc     |0.7609|±  |0.0088|
|             |       |none  |     0|acc_norm|0.7441|±  |0.0090|
|boolq        |Yaml   |none  |     0|acc     |0.7780|±  |0.0073|
|hellaswag    |Yaml   |none  |     0|acc     |0.5694|±  |0.0049|
|             |       |none  |     0|acc_norm|0.7549|±  |0.0043|
|piqa         |Yaml   |none  |     0|acc     |0.7835|±  |0.0096|
|             |       |none  |     0|acc_norm|0.7911|±  |0.0095|
|winogrande   |Yaml   |none  |     0|acc     |0.6969|±  |0.0129|


CUDA_VISIBLE_DEVICES=6 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=llama-2-7b-silu-full-param-llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
修改hugface代码后再跑一遍relu：用原始的read config办法读取的模型。
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2295|±  |0.0123|
|             |       |none  |     0|acc_norm|0.2585|±  |0.0128|
|arc_easy     |Yaml   |none  |     0|acc     |0.4924|±  |0.0103|
|             |       |none  |     0|acc_norm|0.4482|±  |0.0102|
|boolq        |Yaml   |none  |     0|acc     |0.6257|±  |0.0085|
|hellaswag    |Yaml   |none  |     0|acc     |0.3800|±  |0.0048|
|             |       |none  |     0|acc_norm|0.4829|±  |0.0050|
|piqa         |Yaml   |none  |     0|acc     |0.6752|±  |0.0109|
|             |       |none  |     0|acc_norm|0.6872|±  |0.0108|
|winogrande   |Yaml   |none  |     0|acc     |0.5588|±  |0.0140|

无微调的训练情况：relu
CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --model_args pretrained=llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2432|±  |0.0125|
|             |       |none  |     0|acc_norm|0.2892|±  |0.0133|
|arc_easy     |Yaml   |none  |     0|acc     |0.2795|±  |0.0092|
|             |       |none  |     0|acc_norm|0.2921|±  |0.0093|
|boolq        |Yaml   |none  |     0|acc     |0.5991|±  |0.0086|
|hellaswag    |Yaml   |none  |     0|acc     |0.2582|±  |0.0044|
|             |       |none  |     0|acc_norm|0.2600|±  |0.0044|
|piqa         |Yaml   |none  |     0|acc     |0.5408|±  |0.0116|
|             |       |none  |     0|acc_norm|0.5218|±  |0.0117|
|winogrande   |Yaml   |none  |     0|acc     |0.4988|±  |0.0141|

无微调训练情况：silu
CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --model_args pretrained=llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.4326|±  |0.0145|
|             |       |none  |     0|acc_norm|0.4616|±  |0.0146|
|arc_easy     |Yaml   |none  |     0|acc     |0.7643|±  |0.0087|
|             |       |none  |     0|acc_norm|0.7458|±  |0.0089|
|boolq        |Yaml   |none  |     0|acc     |0.7810|±  |0.0072|
|hellaswag    |Yaml   |none  |     0|acc     |0.5715|±  |0.0049|
|             |       |none  |     0|acc_norm|0.7598|±  |0.0043|
|piqa         |Yaml   |none  |     0|acc     |0.7797|±  |0.0097|
|             |       |none  |     0|acc_norm|0.7911|±  |0.0095|
|winogrande   |Yaml   |none  |     0|acc     |0.6930|±  |0.0130|

silu微调的测试结果：
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=llama-2-7b-silu-full-param-llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.4326|±  |0.0145|
|             |       |none  |     0|acc_norm|0.4497|±  |0.0145|
|arc_easy     |Yaml   |none  |     0|acc     |0.7668|±  |0.0087|
|             |       |none  |     0|acc_norm|0.7445|±  |0.0089|
|boolq        |Yaml   |none  |     0|acc     |0.8128|±  |0.0068|
|hellaswag    |Yaml   |none  |     0|acc     |0.5577|±  |0.0050|
|             |       |none  |     0|acc_norm|0.7582|±  |0.0043|
|piqa         |Yaml   |none  |     0|acc     |0.7851|±  |0.0096|
|             |       |none  |     0|acc_norm|0.7938|±  |0.0094|
|winogrande   |Yaml   |none  |     0|acc     |0.7009|±  |0.0129|

relu微调的测试结果：
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2295|±  |0.0123|
|             |       |none  |     0|acc_norm|0.2585|±  |0.0128|
|arc_easy     |Yaml   |none  |     0|acc     |0.4924|±  |0.0103|
|             |       |none  |     0|acc_norm|0.4482|±  |0.0102|
|boolq        |Yaml   |none  |     0|acc     |0.6257|±  |0.0085|
|hellaswag    |Yaml   |none  |     0|acc     |0.3800|±  |0.0048|
|             |       |none  |     0|acc_norm|0.4829|±  |0.0050|
|piqa         |Yaml   |none  |     0|acc     |0.6752|±  |0.0109|
|             |       |none  |     0|acc_norm|0.6872|±  |0.0108|
|winogrande   |Yaml   |none  |     0|acc     |0.5588|±  |0.0140|

## 训练
感觉不太对劲啊！用silu微调后的模型重新在relu情况下train一遍：
CUDA_VISIBLE_DEVICES=4,5 torchrun --nnodes 1 --nproc_per_node 2 examples/finetuning_tuned_models.py --enable_fsdp --model_name checkpoint/llama-2-7b-hf --fsdp checkpoint/llama-2-7b-silu-full-param-llama-2-7b-hf --dist_checkpoint_root_folder checkpoint --dist_checkpoint_folder relu-full-param-after-silu-tuned

改用refinedweb_dataset来训练
CUDA_VISIBLE_DEVICES=0,4,5 torchrun --nnodes 1 --nproc_per_node 3 examples/finetuning_tuned_models.py --dataset "refinedweb_dataset" --enable_fsdp --model_name checkpoint/llama-2-7b-hf --dist_checkpoint_root_folder checkpoint --dist_checkpoint_folder silu-full-param-refinedweb
这个代码跑两个晚上都没跑完，数据集下载还没结束，大概率有问题。


1.5:继续测试修改回原代码后的lm-eval，找到当时如何计算得到的数据。
CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=llama-2-7b-silu-full-param-llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16

微调恢复模型的测试：
CUDA_VISIBLE_DEVICES=7 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=relu-full-param-after-silu-tuned-checkpoint/llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2304|±  |0.0123|
|             |       |none  |     0|acc_norm|0.2841|±  |0.0132|
|arc_easy     |Yaml   |none  |     0|acc     |0.2748|±  |0.0092|
|             |       |none  |     0|acc_norm|0.2820|±  |0.0092|
|boolq        |Yaml   |none  |     0|acc     |0.5957|±  |0.0086|
|hellaswag    |Yaml   |none  |     0|acc     |0.2577|±  |0.0044|
|             |       |none  |     0|acc_norm|0.2610|±  |0.0044|
|piqa         |Yaml   |none  |     0|acc     |0.5354|±  |0.0116|
|             |       |none  |     0|acc_norm|0.5114|±  |0.0117|
|winogrande   |Yaml   |none  |     0|acc     |0.4917|±  |0.0141|
效果仍然不佳


几个问题：
1. relu层微调后恢复不行。详细描述是：采用llama模型、自带数据集微调、学习率为论文提供、测试任务与论文一致、
其他超参为默认的情况下，只做论文中的阶段一部分（silu改为relu）。
结果表明：
silu微调结果与论文基本一致；relu改后性能确实下降；但是relu微调后性能没有恢复。
   可能出错地方：
   eval的时候，采用了错误的load方式？load进来的模型与理解的relu微调模型不一致
   微调的时候只改了config，理解和实际的改动不同？导致测试结果与实验描述不一致。
   微调的时候会影响llama-2-7B-hf这个ckpt的参数？导致新的微调会改变旧的微调结果吗
   可能：dataset太小了（train得更充分）samsum一小时以内就能跑完。

2. refinedweb使用论文中说明的数据集的时候，训练过程一直在加载代码。应该是有问题的。

3. relu层计算激活比例的效果差别不大。在测试的里面做的。

趁着relu计算的代码也写完了，重新训练：
1. silu在原llama-2模型上微调；
2. relu在原llama-2模型上微调。
先做这两个。
# 训练1.
CUDA_VISIBLE_DEVICES=0,1,4,5,6 torchrun --nnodes 1 --nproc_per_node 5 examples/finetuning_tuned_models.py --enable_fsdp --model_name checkpoint/llama-2-7b-hf --dist_checkpoint_root_folder checkpoint --dist_checkpoint_folder silu-full-param-retune-with-zero-ratio --output_dir /disk3/Haonan/yanbo_random/llama-recipes/matrics/silu-full-param-retune-with-zero-ratio

# 训练2
CUDA_VISIBLE_DEVICES=0,1,4,5,6 torchrun --nnodes 1 --nproc_per_node 5 examples/finetuning_tuned_models.py --enable_fsdp --model_name checkpoint/llama-2-7b-hf --dist_checkpoint_root_folder checkpoint --dist_checkpoint_folder relu-full-param-retune-with-zero-ratio --output_dir /disk3/Haonan/yanbo_random/llama-recipes/matrics/relu-full-param-retune-with-zero-ratio

1,6; 0,1,4,5,6
# 测试2
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=relu-full-param-retune-with-zero-ratio-checkpoint/llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |     0|acc     |0.2304|±  |0.0123|
|             |       |none  |     0|acc_norm|0.2841|±  |0.0132|
|arc_easy     |Yaml   |none  |     0|acc     |0.2748|±  |0.0092|
|             |       |none  |     0|acc_norm|0.2820|±  |0.0092|
|boolq        |Yaml   |none  |     0|acc     |0.5957|±  |0.0086|
|hellaswag    |Yaml   |none  |     0|acc     |0.2577|±  |0.0044|
|             |       |none  |     0|acc_norm|0.2610|±  |0.0044|
|piqa         |Yaml   |none  |     0|acc     |0.5354|±  |0.0116|
|             |       |none  |     0|acc_norm|0.5114|±  |0.0117|
|winogrande   |Yaml   |none  |     0|acc     |0.4917|±  |0.0141|
# 测试1
CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --model_args pretrained=llama-2-7b-hf,fsdp=silu-full-param-retune-with-zero-ratio-checkpoint/llama-2-7b-hf --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --batch_size 16


# 修改了一下微调时读入模型的方法。
再次训练：
CUDA_VISIBLE_DEVICES=4,6 torchrun --nnodes 1 --nproc_per_node 2 examples/finetuning_tuned_models.py --enable_fsdp --model_name checkpoint/llama-2-7b-hf --dist_checkpoint_root_folder checkpoint --dist_checkpoint_folder relu-full-param-retune-with-conf-load --output_dir /disk3/Haonan/yanbo_random/llama-recipes/matrics/relu-full-param-retune-with-conf-load

setting：
1. llama自带的实验设置
   samsum dataset.->1e-4
   epoch 3（可能有问题）


2. 论文设置
   1.refinedweb：测了一下，loaddata func有问题。下载问题/代码确定
   lr: 1e-5
   llama(我用的是2)->finetune->silu
   llama(我用的是2)->finetune->relu
   

测试：
   lm-eval

1. relu



bert
CUDA_VISIBLE_DEVICES=1 python train.py
{'loss': 1.3397, 'learning_rate': 9e-06, 'epoch': 1.0}
 10%|████████                                                                        | 500/5000 [00:53<07:36,  9.85it/s]Checkpoint destination directory /disk3/Haonan/yanbo_random/bert_finetune/checkpoint/checkpoint-500 already exists and is non-empty.Saving will proceed but saved results may be invalid.
{'eval_loss': 1.0086137056350708, 'eval_runtime': 8.0012, 'eval_samples_per_second': 124.981, 'eval_steps_per_second': 15.623, 'epoch': 1.0}

{'loss': 0.9124, 'learning_rate': 8.000000000000001e-06, 'epoch': 2.0}
{'eval_loss': 1.02223801612854, 'eval_runtime': 7.9532, 'eval_samples_per_second': 125.735, 'eval_steps_per_second': 15.717, 'epoch': 2.0}

{'loss': 0.6519, 'learning_rate': 7e-06, 'epoch': 3.0}
{'eval_loss': 1.0733144283294678, 'eval_runtime': 7.925, 'eval_samples_per_second': 126.184, 'eval_steps_per_second': 15.773, 'epoch': 3.0}

{'loss': 0.4483, 'learning_rate': 6e-06, 'epoch': 4.0}
{'eval_loss': 1.377542495727539, 'eval_runtime': 7.9155, 'eval_samples_per_second': 126.334, 'eval_steps_per_second': 15.792, 'epoch': 4.0}

{'loss': 0.3062, 'learning_rate': 5e-06, 'epoch': 5.0}
{'eval_loss': 1.5806139707565308, 'eval_runtime': 7.9632, 'eval_samples_per_second': 125.577, 'eval_steps_per_second': 15.697, 'epoch': 5.0}

{'loss': 0.209, 'learning_rate': 4.000000000000001e-06, 'epoch': 6.0}
{'eval_loss': 1.8742899894714355, 'eval_runtime': 7.9455, 'eval_samples_per_second': 125.857, 'eval_steps_per_second': 15.732, 'epoch': 6.0}

{'loss': 0.1382, 'learning_rate': 3e-06, 'epoch': 7.0}
{'eval_loss': 2.1105427742004395, 'eval_runtime': 7.9281, 'eval_samples_per_second': 126.134, 'eval_steps_per_second': 15.767, 'epoch': 7.0}

{'loss': 0.0992, 'learning_rate': 2.0000000000000003e-06, 'epoch': 8.0}
{'eval_loss': 2.2459874153137207, 'eval_runtime': 7.9657, 'eval_samples_per_second': 125.538, 'eval_steps_per_second': 15.692, 'epoch': 8.0}

{'loss': 0.0701, 'learning_rate': 1.0000000000000002e-06, 'epoch': 9.0}
{'eval_loss': 2.2909467220306396, 'eval_runtime': 7.9342, 'eval_samples_per_second': 126.037, 'eval_steps_per_second': 15.755, 'epoch': 9.0}

{'loss': 0.0631, 'learning_rate': 0.0, 'epoch': 10.0}
{'eval_loss': 2.3056459426879883, 'eval_runtime': 7.9143, 'eval_samples_per_second': 126.353, 'eval_steps_per_second': 15.794, 'epoch': 10.0}

{'train_runtime': 606.6277, 'train_samples_per_second': 32.969, 'train_steps_per_second': 8.242, 'train_loss': 0.4238158580780029, 'epoch': 10.0}

(llamaRelu) dltp_haonan@dgx:/disk3/Haonan/yanbo_random/bert_finetune$
训练效果看起来不够理想。
增加训练集大小。

dataset: yelp_review_full
set: 100000
CUDA_VISIBLE_DEVICES=1,5 python train.py
serWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
{'loss': 0.9228, 'learning_rate': 4.065040650406504e-05, 'epoch': 1.0}
{'eval_loss': 0.9020474553108215, 'eval_accuracy': 0.61015, 'eval_runtime': 104.9193, 'eval_samples_per_second': 190.623, 'eval_steps_per_second': 11.914, 'epoch': 1.0}                                                                                                          | 6250/31250 [25:09<1:31:42,  4.54it/s/home/dltp_haonan/anaconda3/envs/llamaRelu/lib/python3.9/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
{'loss': 0.7307, 'learning_rate': 3.048780487804878e-05, 'epoch': 2.0}
{'eval_loss': 0.8959066271781921, 'eval_accuracy': 0.64195, 'eval_runtime': 104.0966, 'eval_samples_per_second': 192.129, 'eval_steps_per_second': 12.008, 'epoch': 2.0}
{'loss': 0.5675, 'learning_rate': 2.032520325203252e-05, 'epoch': 3.0}
{'eval_loss': 0.8646249175071716, 'eval_accuracy': 0.6591, 'eval_runtime': 129.4053, 'eval_samples_per_second': 154.553, 'eval_steps_per_second': 9.66, 'epoch': 3.0}
{'loss': 0.38, 'learning_rate': 1.016260162601626e-05, 'epoch': 4.0}
{'eval_loss': 1.070185661315918, 'eval_accuracy': 0.65335, 'eval_runtime': 191.9662, 'eval_samples_per_second': 104.185, 'eval_steps_per_second': 6.512, 'epoch': 4.0}
{'loss': 0.2253, 'learning_rate': 0.0, 'epoch': 5.0}
{'eval_loss': 1.4374538660049438, 'eval_accuracy': 0.6522, 'eval_runtime': 192.016, 'eval_samples_per_second': 104.158, 'eval_steps_per_second': 6.51, 'epoch': 5.0}
{'train_runtime': 9667.2524, 'train_samples_per_second': 51.721, 'train_steps_per_second': 3.233, 'train_loss': 0.56525721484375, 'epoch': 5.0}

1000
CUDA_VISIBLE_DEVICES=3,6 python test.py
{'accuracy': 0.594}
{'accuracy': 0.635}
{'accuracy': 0.661} better
{'accuracy': 0.65}
{'accuracy': 0.646}

relu ft:
train: 1000
CUDA_VISIBLE_DEVICES=1,6 python train.py
{'loss': 1.5199, 'learning_rate': 6.300000000000001e-06, 'epoch': 1.0}
{'eval_loss': 1.2713137865066528, 'eval_accuracy': 0.49735, 'eval_runtime': 183.0257, 'eval_samples_per_second': 109.274, 'eval_steps_per_second': 6.83, 'epoch': 1.0}
{'loss': 0.9538, 'learning_rate': 1.2600000000000001e-05, 'epoch': 2.0}
{'eval_loss': 0.9104380011558533, 'eval_accuracy': 0.61875, 'eval_runtime': 178.982, 'eval_samples_per_second': 111.743, 'eval_steps_per_second': 6.984, 'epoch': 2.0}    
{'loss': 0.5978, 'learning_rate': 1.8900000000000002e-05, 'epoch': 3.0}
{'eval_loss': 0.9161300659179688, 'eval_accuracy': 0.642, 'eval_runtime': 183.6381, 'eval_samples_per_second': 108.91, 'eval_steps_per_second': 6.807, 'epoch': 3.0}
{'loss': 0.3399, 'learning_rate': 2.5200000000000003e-05, 'epoch': 4.0}
{'eval_loss': 1.097126841545105, 'eval_accuracy': 0.64345, 'eval_runtime': 179.8147, 'eval_samples_per_second': 111.226, 'eval_steps_per_second': 6.952, 'epoch': 4.0}
{'loss': 0.1906, 'learning_rate': 3.15e-05, 'epoch': 5.0}
{'eval_loss': 1.5229119062423706, 'eval_accuracy': 0.6326, 'eval_runtime': 183.674, 'eval_samples_per_second': 108.889, 'eval_steps_per_second': 6.806, 'epoch': 5.0}
{'train_runtime': 1054.4616, 'train_samples_per_second': 4.742, 'train_steps_per_second': 0.299, 'train_loss': 0.7204125147017222, 'epoch': 5.0}

test 1000 
CUDA_VISIBLE_DEVICES=1,6 python test.py
{'accuracy': 0.487}
{'accuracy': 0.615}
{'accuracy': 0.646}
{'accuracy': 0.641}
{'accuracy': 0.624}
很明显恢复了，但是精度差不到两个点。可能需要等量大小的数据集来train。
算一下sparse吧。

gelu:
sparsity+train 100000
CUDA_VISIBLE_DEVICES=1,6 python train.py
{'loss': 0.9587, 'learning_rate': 4.032258064516129e-05, 'epoch': 1.0}
{'eval_loss': 0.8581370115280151, 'eval_accuracy': 0.62545, 'eval_runtime': 197.6211, 'eval_samples_per_second': 101.204, 'eval_steps_per_second': 12.65, 'epoch': 1.0}
{'loss': 0.7982, 'learning_rate': 3.024193548387097e-05, 'epoch': 2.0}
{'eval_loss': 0.8875149488449097, 'eval_accuracy': 0.63145, 'eval_runtime': 197.7153, 'eval_samples_per_second': 101.156, 'eval_steps_per_second': 12.644, 'epoch': 2.0}
{'loss': 0.6664, 'learning_rate': 2.0161290322580645e-05, 'epoch': 3.0}
{'eval_loss': 0.8722862601280212, 'eval_accuracy': 0.645, 'eval_runtime': 197.5624, 'eval_samples_per_second': 101.234, 'eval_steps_per_second': 12.654, 'epoch': 3.0}
{'loss': 0.5079, 'learning_rate': 1.0080645161290323e-05, 'epoch': 4.0}
{'eval_loss': 1.0274174213409424, 'eval_accuracy': 0.64795, 'eval_runtime': 197.7386, 'eval_samples_per_second': 101.144, 'eval_steps_per_second': 12.643, 'epoch': 4.0}
{'loss': 0.3774, 'learning_rate': 0.0, 'epoch': 5.0}
{'eval_loss': 1.5281929969787598, 'eval_accuracy': 0.645, 'eval_runtime': 197.634, 'eval_samples_per_second': 101.197, 'eval_steps_per_second': 12.65, 'epoch': 5.0}
{'train_runtime': 13836.9091, 'train_samples_per_second': 36.135, 'train_steps_per_second': 4.517, 'train_loss': 0.661729578125, 'epoch': 5.0}

CUDA_VISIBLE_DEVICES=1,3 python test.py
{'accuracy': 0.624}
{'accuracy': 0.631}
{'accuracy': 0.648}
{'accuracy': 0.649}
{'accuracy': 0.64}
测一下精度。
ok的话继续train relu恢复的情况。
relu:
{'loss': 0.5767, 'learning_rate': 4.065040650406504e-05, 'epoch': 1.0}
{'eval_loss': 1.0184783935546875, 'eval_accuracy': 0.6332, 'eval_runtime': 382.7874, 'eval_samples_per_second': 52.248, 'eval_steps_per_second': 3.266, 'epoch': 1.0}
{'eval_loss': 1.123990774154663, 'eval_accuracy': 0.64095, 'eval_runtime': 382.5865, 'eval_samples_per_second': 52.276, 'eval_steps_per_second': 3.267, 'epoch': 2.0}
{'loss': 0.298, 'learning_rate': 2.032520325203252e-05, 'epoch': 3.0}
{'eval_loss': 1.2362258434295654, 'eval_accuracy': 0.64565, 'eval_runtime': 382.745, 'eval_samples_per_second': 52.254, 'eval_steps_per_second': 3.266, 'epoch': 3.0}
{'loss': 0.2305, 'learning_rate': 1.016260162601626e-05, 'epoch': 4.0}
{'eval_loss': 1.6752703189849854, 'eval_accuracy': 0.6434, 'eval_runtime': 387.7429, 'eval_samples_per_second': 51.581, 'eval_steps_per_second': 3.224, 'epoch': 4.0}
{'eval_loss': 2.1670122146606445, 'eval_accuracy': 0.6482, 'eval_runtime': 388.7342, 'eval_samples_per_second': 51.449, 'eval_steps_per_second': 3.216, 'epoch': 5.0}
{'train_runtime': 21074.8514, 'train_samples_per_second': 23.725, 'train_steps_per_second': 1.483, 'train_loss': 0.333551568359375, 'epoch': 5.0}
test: relu完全恢复了！
{'accuracy': 0.64}
{'accuracy': 0.659}
{'accuracy': 0.641}
{'accuracy': 0.652}
{'accuracy': 0.653}

但是记录的sparsity有点问题，怎么iteration都没变化的。
iteration 有问题，记住只用自己层的

基本解决方案，重用llama的训练函数，不用transformer里面带有的trainer
CUDA_VISIBLE_DEVICES=5,6 python src/train.py
现在是2,7

# 首次自己的代码进入多gpu形态
CUDA_VISIBLE_DEVICES=2,7 torchrun --nnodes 1 --nproc_per_node 2 src/train.py
成功了，但是问题很大，太慢了。
之前十万条，3个小时
现在，一万条，五个小时。
大概慢了20倍。

解决了，dataloader放数据到gpu的问题。

gelu:
CUDA_VISIBLE_DEVICES=2,7 torchrun --nnodes 1 --nproc_per_node 2 src/train.py
Key: avg_train_prep, Value: 1.7335594177246094
Key: avg_train_loss, Value: 0.5244717270135879
Key: avg_eval_prep, Value: 2.9751287937164306
Key: avg_eval_loss, Value: 0.8018013834953308
Key: avg_epoch_time, Value: 1348.2525590846315
Key: avg_checkpoint_time, Value: 0.2682927967980504
Key: train_accu, Value: [0.61513, 0.71458, 0.79601, 0.86232, 0.90955]
Key: val_accu, Value: [0.6542, 0.6563, 0.6442, 0.6216, 0.6365]

test:
{'accuracy': 0.627}

config: gelu->relu
{'accuracy': 0.195}

gelu->relu
retrain to recover
Key: avg_train_prep, Value: 1.677800476551056
Key: avg_train_loss, Value: 0.5051478296518326
Key: avg_eval_prep, Value: 2.9986831545829773
Key: avg_eval_loss, Value: 0.853204607963562
Key: avg_epoch_time, Value: 761.8977389025968
Key: avg_checkpoint_time, Value: 1.3582688139285892
Key: train_accu, Value: [0.68388, 0.76322, 0.82805, 0.88148]
Key: val_accu, Value: [0.6415, 0.6405, 0.6309, 0.6143]
test:
{'accuracy': 0.626}
恢复！


# 考虑到我们需要对ctx进行操作，这里跑一下randAD试试ctx是什么结构。
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0000-project --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=0.1 --bootstrap_train=True
在bizon上运行成功了。实际上ctx是无法从外部访问的。

# 用backrazor作为参考，实现了一些稀疏功能，做实验。

checkpoint位置改为sparsify
实验数据量大小改为100
使用的是gelu配置，以及初始ckpt
ok了老铁们
CUDA_VISIBLE_DEVICES=0,2 torchrun --nnodes 1 --nproc_per_node 2 /disk3/Haonan/yanbo_random/bert_finetune/src/train.py
Key: avg_train_prep, Value: 1.8644703328609467
Key: avg_train_loss, Value: 0.6058000177145004
Key: avg_eval_prep, Value: 2.7050459384918213
Key: avg_eval_loss, Value: 0.7914759516716003
Key: avg_epoch_time, Value: 770.8354355068877
Key: avg_checkpoint_time, Value: 1.558990063611418
Key: train_accu, Value: [0.6146, 0.7127, 0.79052, 0.85129]
Key: val_accu, Value: [0.6573, 0.6513, 0.645, 0.6151]
Key: avg_train_prep, Value: 1.8644703328609467
Key: avg_train_loss, Value: 0.6058000177145004
Key: avg_eval_prep, Value: 2.7050459384918213
Key: avg_eval_loss, Value: 0.7914759516716003
Key: avg_epoch_time, Value: 770.3038573462982
Key: avg_checkpoint_time, Value: 1.5368857742287219
Key: train_accu, Value: [0.61659, 0.71321, 0.79044, 0.85187]
Key: val_accu, Value: [0.6573, 0.6513, 0.645, 0.6151]
只有kqv
接下来全部替换
Key: avg_train_prep, Value: 2.256462901830673
Key: avg_train_loss, Value: 0.8082985877990723
Key: avg_eval_prep, Value: 2.409238040447235
Key: avg_train_prep, Value: 2.256462901830673
Key: avg_eval_loss, Value: 0.8692525774240494
Key: avg_train_loss, Value: 0.8082985877990723
Key: avg_epoch_time, Value: 954.5768471234478
Key: avg_eval_prep, Value: 2.409238040447235
Key: avg_checkpoint_time, Value: 1.4602470679674298
Key: avg_eval_loss, Value: 0.8692525774240494
Key: avg_epoch_time, Value: 955.5093130024616
Key: train_accu, Value: [0.56965, 0.64123, 0.67479, 0.70644]
Key: avg_checkpoint_time, Value: 1.428373244823888
Key: val_accu, Value: [0.5999, 0.6173, 0.6222, 0.6234]
Key: train_accu, Value: [0.57194, 0.63904, 0.67565, 0.70642]
Key: val_accu, Value: [0.5999, 0.6173, 0.6222, 0.6234]
可以明显发现都变成稀疏之后，epoch不够了。
epoch10
Epoch 10: train_perplexity=1.5466, train_epoch_loss=0.4361, epoch time 945.6940981615335s
training params are saved in /disk3/Haonan/yanbo_random/bert_finetune/checkpoint/train_basic_yelp_gelu_100000_sparsify_1/train_params.yaml
Key: avg_train_prep, Value: 1.916380262374878
Key: avg_train_loss, Value: 0.6364706039428711
Key: avg_eval_prep, Value: 2.6851329803466797
Key: avg_eval_loss, Value: 0.8698895454406739
Key: avg_epoch_time, Value: 972.1122430929914
Key: avg_checkpoint_time, Value: 1.1579198642633854
Key: train_accu, Value: [0.57025, 0.63923, 0.67461, 0.70766, 0.73769, 0.76327, 0.78346, 0.80041, 0.8166, 0.8284]
Key: val_accu, Value: [0.5962, 0.6147, 0.6248, 0.6225, 0.6212, 0.6194, 0.62, 0.6189, 0.6188, 0.6154]
Key: avg_train_prep, Value: 1.916380262374878
Key: avg_train_loss, Value: 0.6364706039428711
Key: avg_eval_prep, Value: 2.6851329803466797
Key: avg_eval_loss, Value: 0.8698895454406739
Key: avg_epoch_time, Value: 972.9308409936726
Key: avg_checkpoint_time, Value: 1.1418508382514119
Key: train_accu, Value: [0.56838, 0.64038, 0.67582, 0.70693, 0.73795, 0.76274, 0.7851, 0.80043, 0.81765, 0.82786]
Key: val_accu, Value: [0.5962, 0.6147, 0.6248, 0.6225, 0.6212, 0.6194, 0.62, 0.6189, 0.6188, 0.6154]

加了双向乘法后变得特别慢
time_period_1:  1.3130716979503632e-05
time_period_2:  0.00012634042650461197
time_period_3:  5.775783210992813e-05
time_period_4:  0.23104449082165956
time_period_5:  9.00961458683014e-06
time_period_6:  4.0786340832710266e-05
主要是第四步
就是恢复的过程，该过程不可省，而且超级慢。
必须对其他维度也作稀疏！
Key: avg_train_prep, Value: 2.7400333642959596
Key: avg_train_loss, Value: 1.00632786154747
Key: avg_eval_prep, Value: 2.642280030250549
Key: avg_eval_loss, Value: 0.9713226616382599
Key: avg_epoch_time, Value: 1163.6173834130168
Key: avg_checkpoint_time, Value: 1.31673616534099
Key: train_accu, Value: [0.48413, 0.54539, 0.55587, 0.56289, 0.5654, 0.56865, 0.57119, 0.57254, 0.57201, 0.57316]
Key: val_accu, Value: [0.544, 0.5623, 0.5711, 0.5706, 0.5731, 0.5764, 0.5763, 0.5764, 0.5765, 0.577]
Key: avg_train_prep, Value: 2.7400333642959596
Key: avg_train_loss, Value: 1.00632786154747
Key: avg_eval_prep, Value: 2.642280030250549
Key: avg_eval_loss, Value: 0.9713226616382599
Key: avg_epoch_time, Value: 1163.554242487345
Key: avg_checkpoint_time, Value: 1.3719362079165875
Key: train_accu, Value: [0.48292, 0.54729, 0.55629, 0.5617, 0.56417, 0.57027, 0.57057, 0.57198, 0.57217, 0.5741]
Key: val_accu, Value: [0.544, 0.5623, 0.5711, 0.5706, 0.5731, 0.5764, 0.5763, 0.5764, 0.5765, 0.577]

采用sparse tensor作了改造，现在只做最后一维稀疏！
train_yelp_gelu_100000_sparsify_3
Key: avg_train_prep, Value: 1.8442419528961183
Key: avg_train_loss, Value: 0.5948109328746796
Key: avg_eval_prep, Value: 2.7120981216430664
Key: avg_eval_loss, Value: 0.7917495965957642
Key: avg_epoch_time, Value: 1478.1142883392051
Key: avg_checkpoint_time, Value: 1.0754049364477396
Key: train_accu, Value: [0.6107, 0.69577, 0.76001, 0.81497, 0.85858]
Key: val_accu, Value: [0.6514, 0.6511, 0.6469, 0.619, 0.6305]
Key: avg_train_prep, Value: 1.8442419528961183
Key: avg_train_loss, Value: 0.5948109328746796
Key: avg_eval_prep, Value: 2.7120981216430664
Key: avg_eval_loss, Value: 0.7917495965957642
Key: avg_epoch_time, Value: 1478.1097237305717
Key: avg_checkpoint_time, Value: 1.0655514124780894
Key: train_accu, Value: [0.60956, 0.69482, 0.76067, 0.81381, 0.85787]
Key: val_accu, Value: [0.6514, 0.6511, 0.6469, 0.619, 0.6305]
效果碉堡了

整理一下师兄的代码写法
1. expand
forward存成一个slice

sparse * dense = dense
dataloader这里尽量截断一下。
backward用cpu测。这样比较快，怎么简洁怎么来。
一张卡也挺快的，bert，没必要多卡train
测一下每个iteration的速度。
我可能测得再细一点，在里面测一下每个计算的速度也好。
batch[keys] -> to.device，有可能会导致代码速度慢
dataloader要小心一下
看看star比较多的repo。
直接搜yelp review full这个dataloader的写法。
loss.item简化一下。

sparse * dense 用bmm，肯定能成
四维的这种乘法有可能需要reshape到三维。
reshape操作。
多用torch本身操作，相对clean一点。
相乘得到dense的话，就不必用sparse adam了
这个新写法速度没理由更慢。
对比一下二者的速度。
只要用put或者gather就特别慢。大概bizon五十多秒。

SPAR，做了一些inplace操作，尽量省memory
现在尽量把额外的性能都榨一榨。


都写平了速度和backrazor比一下。
有没有其他性能可以榨取的。
其他操作符可以写的？ backrazor，和他们对齐，都用sparse的做法。
relu，gelu inplace搞一搞
random shaffle是最朴素的方法。
用合理的方式去select column
一个靠norm，一个靠random，一个靠variance

尽量接近他们的操作符，找可以提升的点。
Tempo: Accelerating Transformer-Based Model Training through Memory Footprint Reduction (Tempo: (1) In-place GELU, (2) In-place LayerNorm, and (3) Sub-Layer Dropout Recomputation.)
能搬运的搬运，能改造最好。尽可能的榨取性能。

1.按照上述修改代码，并且搬运到bizon上面。
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 /home/bizon/yanbo_random/bert_finetune_sparsify/src/train.py
注意这里load data速度很慢
代码跑应该是正确的了，但是out of mem了，解决办法是先减小batch size
估计每个epoch大概45分钟，这个速度还是可以的。
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 /home/bizon/yanbo_random/bert_finetune_sparsify_new_version/src/train.py
试试这个！

需要根据backRazor增加的新稀疏层。
BertIntermediate
BertOutput
BertPooler
BertForSequenceClassification
BertSelfAttention
BertSelfOutput

ok了家人们，把norm初步实现了一下啊，看看对不对。

初步实现了，但是在bizon服务器上跑的效果确实有点烂，先用非稀疏的把基线给重跑一下。
然后检查哪个组件添加之后性能掉的最厉害，说明有问题。
CUDA_VISIBLE_DEVICES=7 torchrun --nnodes 1 --nproc_per_node 1 /disk3/Haonan/yanbo_random/bert_finetune_sparsify/src/train.py

测试完成了。现在已知修改的几个稀疏组件都不会严重降低性能，是正确的。
代码重构：
1.去掉多gpu功能，给训练过程加速
2.去掉transformers功能，彻底给代码减重。

现在减重也已经完成了。

准备和randAD或者backRazor比较一下。

主要比较：
1. flops 计算量
2. macs 访存量
3. memory 内存占用
4. time 训练时间
CUDA_VISIBLE_DEVICES=7 python src/train.py
初步使用对象化有问题:内存占用太高,两个稀疏占了58635多.
原本的也不遑多让: 63350MiB
三个linear:60937MiB

总结的一些经验:
forward需要越快越好,千万不能用一些复杂逻辑.
做sparse的时候,甚至可以考虑

require grad函数会复制值,产生mem
尽可能就地操作减少内存.

数据集可以删减，但是可以多跑几个数据集，避免overfit

if else 太多，重用的代码少，分开写清楚，不容易混乱。
有必要的话拆成两个file


时间、内存、精度 （星状图，战力图）
1. performance 还正常，降低了几个点
2. memory norm 有点高于rand， softmax和layer norm
3. 速度需要比较一下正常的
4. 

可提升的方法：
inplace tempo layernorm
inplace tempo gelu

1. 想办法根据三个指标，提升方法的效果
2. 着重考虑inplace的方法，提升内存。
3. 测一下baseline，backrazor/randAD/noSparse。
4. 修改relu和gelu的就地操作。
每个部分有个开关。这可以看到每个部分对实验的影响。看看是那里导致的内存爆炸。
先no sparse，
再加一部分layer的稀疏，看看它们带来的变化
1）linear有提示，（除了kq那里都改了。）
2）matmul k*q
3）softmax matmul kq*v
4) layer norm 
5) activation
先跑它们的方法，再贴近我们的方法。
forward全量乘，column 和row记录下来，不用全求。

performance测起来超过no sparse。

编译错误：
在conda里面装，用一个新的conda环境也行。其实自己的环境就可以了。
conda install 一个c的版本。一般能解决
或者有link错误，手动改clang。一个指向错误。
python里面说c的问题，一般都是conda install版本。

第一阶段：idea实现
第二阶段：严谨实验，设计实验，对比性能
第三阶段：paper

