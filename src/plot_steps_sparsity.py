import matplotlib.pyplot as plt
import pandas as pd
# 注意修改读入的txt文件名称以及生成的图像名称，以防止加载错误或者误覆盖。
def plot_layer_with_steps(layer_idx, color = "blue", islabel = False, act_type = 'gelu',isSmooth = False, gpu = 2):
    assert act_type in ['gelu','relu']
    # layer_steps_ratio = {'step':[],'ratio':[]}
    layer_steps_ratio = {}
    with open(f'/disk3/Haonan/yanbo_random/bert_finetune/zero_ratio/zero-ratio-of-bert-layer-{act_type}-training.txt', 'r') as file:
        for line in file:
            if f'layer{layer_idx}:' in line:
                step = int(line.split("\n")[0].split(';')[0].split(':')[1])
                ratio = float(line.split("\n")[0].split(';')[1].split(':')[1])
                if step not in layer_steps_ratio:
                    layer_steps_ratio[step] = ratio
                else:
                    layer_steps_ratio[step] += ratio
    steps = list(layer_steps_ratio.keys())
    values = [value / gpu for value in layer_steps_ratio.values()] # average
    df = pd.DataFrame({'step':steps,'ratio':values})
    print(df.shape)
    if act_type == 'gelu':
        # epoch 5 train 1563, eval 157 total each epoch 1720, total 8600
        bins = [-1,1562,1719,3282,3439,5002,5159,6722,6879,8442,8599]
        df['run_type']=pd.cut(df['step'], bins=bins, ordered=False, labels=['train','eval','train','eval','train','eval','train','eval','train','eval'])
    elif act_type == 'relu':
        # epoch 4 train 1563, eval 157 total each epoch 1720, total 6880
        bins = [-1,1562,1719,3282,3439,5002,5159,6722,6879]
        df['run_type']=pd.cut(df['step'], bins=bins, ordered=False, labels=['train','eval','train','eval','train','eval','train','eval'])
    df_train = df.query("run_type == 'train'")
    print('before drop',df.shape)
    print('after drop',df_train.shape)
    len_train = df_train.shape[0]
    df_train['step_train'] = range(0,len(df_train))
    if isSmooth:
        df_train['ratio_smooth'] = df_train['ratio'].rolling(window=100).mean()
        plt.plot(df_train['step_train'],df_train['ratio_smooth'], color = color, label = f"layer{layer_idx}", alpha = 0.7)
    else:
        plt.plot(df_train['step_train'],df_train['ratio'], color = color, label = f"layer{layer_idx}", alpha = 0.7)
    if islabel:
        label_x = df_train['step_train'][len_train-1]
        label_y = df_train['ratio'][len_train-1]
        plt.text(label_x, label_y, f"layer{layer_idx}", fontsize=9, verticalalignment='bottom')

# windows 指最后作为样本计算稀疏度的平均值的窗口大小
def plot_layers_sparsity(color = 'blue', islabel = False, act_type = 'gelu',isSmooth = False, windows = 100, gpu=2, threshold=0.6):
    assert act_type in ['gelu','relu']
    layer_ratio = {}
    # 取训练完成时的稀疏度。如果不用测试时的数据，则需要考虑：将训练时最后的一部分数据作为输入，按照layer进行平均。
    with open(f'/disk3/Haonan/yanbo_random/bert_finetune/zero_ratio/zero-ratio-of-bert-layer-{act_type}-training.txt', 'r') as file:
        for line in file:
            if 'iteration:' in line:
                step = int(line.split("\n")[0].split(';')[0].split(':')[1])
                if (act_type == 'gelu' and step in range(8443-windows,8443)) or (act_type == 'relu' and step in range(6722-windows,6722)):
                    layer = int(line.split("\n")[0].split(';')[1].split(':')[0].split('layer')[1])
                    ratio = float(line.split("\n")[0].split(';')[1].split(':')[1])
                    if layer not in layer_ratio:
                        layer_ratio[layer] = ratio
                    else:
                        layer_ratio[layer] += ratio
    layers = list(layer_ratio.keys())
    values = [value / (gpu*windows) for value in layer_ratio.values()] # average gpu有多少个，每层就有多少个副本
    df = pd.DataFrame({'layers':layers,'ratio':values})
    plt.plot(df['layers'],df['ratio'], color = color, label = f"{act_type}-{windows}", alpha = 0.7)
    outstanding_values = df[df['ratio'] > threshold]
    for i, row in outstanding_values.iterrows():
        # 在这些点旁边添加注释
        plt.annotate(f"{row['layers']}", (row['layers'], row['ratio']))

   
   

# # 按照step观察稀疏度在训练时的变化曲线：
# # layers: 86 0-85
# for layer_idx in range(86):
#     plt.figure() 
#     plot_layer_with_steps(layer_idx,'gray',True,'gelu')
#     plot_layer_with_steps(layer_idx,'black',True,'gelu',isSmooth = True)
#     plt.title(f'layer {layer_idx} zero ratio for each step')
#     plt.xlabel("step")
#     plt.ylabel('zero ratio')
#     plt.legend(loc='best')
#     # plt.close()
#     plt.savefig(f'/disk3/Haonan/yanbo_random/bert_finetune/plot/gelu_zero_ratio_layer_{layer_idx}.png')
#     plt.close() 

# # 对比多个layer的稀疏度变化曲线：
# plt.figure() 
# for layer_idx in range(86):
#     if layer_idx %10 == 0:
#         plot_layer_with_steps(layer_idx,'black',True,'gelu',isSmooth = True)
# plt.title(f'layer {layer_idx} zero ratio for each step')
# plt.xlabel("step")
# plt.ylabel('zero ratio')
# plt.legend(loc='best')
# # plt.close()
# plt.savefig(f'/disk3/Haonan/yanbo_random/bert_finetune/plot/gelu_zero_ratio_layers_x0.png')
# plt.close() 

# # relu:
# # layers: 86 0-85
# for layer_idx in range(86):
#     plt.figure() 
#     plot_layer_with_steps(layer_idx,'gray',True,'relu')
#     plot_layer_with_steps(layer_idx,'black',True,'relu',isSmooth = True)
#     plt.title(f'layer {layer_idx} zero ratio for each step')
#     plt.xlabel("step")
#     plt.ylabel('zero ratio')
#     plt.legend(loc='best')
#     # plt.close()
#     plt.savefig(f'/disk3/Haonan/yanbo_random/bert_finetune/plot/relu_zero_ratio_layer_{layer_idx}.png')
#     plt.close() 

# # 对比多个layer的稀疏度变化曲线：
# plt.figure() 
# for layer_idx in range(86):
#     if layer_idx %10 == 0:
#         plot_layer_with_steps(layer_idx,'black',True,'relu',isSmooth = True)
# plt.title(f'layer {layer_idx} zero ratio for each step')
# plt.xlabel("step")
# plt.ylabel('zero ratio')
# plt.legend(loc='best')
# # plt.close()
# plt.savefig(f'/disk3/Haonan/yanbo_random/bert_finetune/plot/relu_zero_ratio_layers_x0.png')
# plt.close() 

# 观察训练完成后每个layer的稀疏度：
plt.figure()
plot_layers_sparsity('gray',True,'gelu')
plot_layers_sparsity('red',True,'relu')
plt.xlabel("layer")
plt.ylabel('zero ratio')
plt.legend(loc='best')
plt.title(f'layers zero ratio')
plt.savefig(f'/disk3/Haonan/yanbo_random/bert_finetune/plot/relu_zero_ratio_all_layers_last_windows_sparsity.png')
plt.close() 
