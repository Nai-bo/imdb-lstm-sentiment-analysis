# 改进版：基于原 Notebook 的小范围受控实验

## 目标

改进版不是重新实现一个复杂项目。它直接沿用原 Notebook 的数据读取、80% / 10% /
10% 划分、文本清理、词表、整数序列、单层单向 LSTM 和主要训练配置，只修改会
影响 Random/GloVe 对照的部分。

改动集中在五点：

1. 固定 Python、NumPy、PyTorch、CUDA 和 DataLoader 随机性；
2. 记录真实 lengths，并使用 `pack_padded_sequence` 排除 padding；
3. Random Embedding 和 GloVe Embedding 都冻结；
4. 每轮根据验证集 Accuracy 保存最佳模型，平手时选择验证 Loss 更低者；
5. 使用 scikit-learn 计算指标，并保存训练历史、Loss 曲线、指标图和 ROC 曲线。

## 实验配置

| 配置 | 值 |
| --- | --- |
| 原始评论 | IMDb 25,000 条训练评论 |
| 训练集 | 20,000 条 |
| 验证集 | 2,500 条 |
| 测试集 | 2,500 条，来自原训练数据的自定义划分 |
| 数据划分 seed | 42 |
| 训练 seed | 42 |
| 词表 | 仅使用训练集构建，最多 10,000 token |
| 特殊 token | `<pad>`、`<unk>` |
| Embedding 维度 | 300 |
| GloVe | `glove.6B.300d.txt`，覆盖率 9,851 / 10,000，即 98.51% |
| LSTM | 单层、单向、hidden size 512 |
| Dropout | 0.5 |
| Batch size | 32 |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam，学习率 0.0008 |
| Epochs | 10 |
| Embedding 状态 | Random 与 GloVe 都冻结 |
| 模型选择 | 验证集 Accuracy 最高；平手看验证 Loss |
| 分类方式 | 二维 logits 的 `argmax` |

完整配置见 [`configs/experiment.json`](configs/experiment.json)。

## 当前结果

本次运行使用 seed 42，只运行一次：

| Embedding | Accuracy | Precision | Recall | F1 | ROC-AUC | 最佳轮次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random，冻结 | 0.8340 | 0.8413 | 0.8273 | 0.8342 | 0.9093 | 7 |
| GloVe，冻结 | 0.8632 | 0.8840 | 0.8391 | 0.8610 | 0.9376 | 7 |

差值：

| 指标 | GloVe - Random |
| --- | ---: |
| Accuracy | +0.0292 |
| Precision | +0.0427 |
| Recall | +0.0119 |
| F1 | +0.0268 |
| ROC-AUC | +0.0283 |

在这组实验中，GloVe 的 Accuracy 比 Random 高 2.92 个百分点，ROC-AUC 高
0.0283。方向与历史简历中“Random 高 7.24 个百分点”相反。

这个变化说明原实验的排名对随机性和实验设计很敏感。它不能单独证明“GloVe
一定优于 Random”，也不能证明“历史结果完全错误”。合理结论是：

> 在原数据划分和当前小范围受控配置下，冻结 GloVe 的测试表现更好；但实验只有
> 一个种子，排序稳定性还需要重复实验才能确认。

## 为什么这样改

### 1. 固定随机性

`seed_everything()` 统一设置：

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- cuDNN deterministic

训练 DataLoader 使用 `torch.Generator().manual_seed(42)`。验证集和测试集关闭
shuffle，保证评估顺序稳定。

### 2. 两组模型都冻结 Embedding

原版中 Random Embedding 可训练，GloVe 冻结，比较同时改变了两件事：

- 初始向量来自随机值还是 GloVe
- Embedding 是否继续更新

改进版把两者都冻结，使主要差异落在初始化来源上。这个选择适合比较“冻结初始化
表示”的效果，但不代表冻结就是追求最高精度的最佳方案。

### 3. 使用 `pack_padded_sequence`

同一 batch 的评论长度不同，因此必须先补齐成矩形张量。padding 只用于组成 batch，
不是真实单词。模型接收每条序列的真实长度后：

```text
整数 ID -> Embedding -> pack_padded_sequence -> LSTM
```

LSTM 不会把 padding 当成真实时间步，最终 hidden state 只由有效 token 计算。

### 4. 按验证集选择 checkpoint

每个 epoch 后计算验证集指标，并保存 Accuracy 最高的模型参数。测试集只在最佳
状态恢复后计算一次，不参与模型选择。这样避免把测试集用于调参，也避免直接使用
可能已经过拟合的最后一轮。

### 5. 使用 scikit-learn 计算指标

Accuracy、Precision、Recall、F1、混淆矩阵和 ROC-AUC 都由 scikit-learn 计算，
减少手写阈值、正负类和除零处理带来的错误。

## 训练曲线说明了什么

Random 的训练 Loss 从第 3 到第 10 轮持续下降，但验证 Loss 从第 4 轮之后明显上升，
说明模型开始过拟合。结论不是简单地选了“验证 Accuracy 最高的第 7 轮”，还要结合
验证 Loss、F1 和 AUC 一起判断。

GloVe 的验证 Loss 在最佳轮次附近较稳定，但第 8 轮后也明显上升。两组最终测试
都恢复验证集 Accuracy 最佳轮次的参数，而不是最后一轮参数。

## 与原版的差异

| 问题 | 原版 | 改进版 |
| --- | --- | --- |
| 随机性 | 历史实验未完整固定；当前可运行版有 seed 42 | 完整固定主要随机源和 DataLoader |
| Embedding 可训练状态 | Random 可训练，GloVe 冻结 | 两者都冻结 |
| Padding | `<pad>` 进入 LSTM | `pack_padded_sequence` |
| 测试模型 | 最后一轮 | 验证集 Accuracy 最佳轮次 |
| 指标 | 手写 | scikit-learn |
| ROC 曲线 | 只保存 AUC 数值 | 保存两条真实 ROC 曲线 |
| 结果文件 | 终端输出和两张图 | 指标 JSON、环境 JSON 和三张图 |

## 目录

```text
improved/
  Lstm_model_train_and_test_improved.ipynb
  README.md
  requirements.txt
  configs/
    experiment.json
    executed_environment.json
  artifacts/
    metrics.json
    executed_environment.json
    loss_curves.png
    metrics_comparison.png
    roc_comparison.png
```

Notebook 训练时会临时生成 `best_random.pt` 和 `best_pretrained.pt`，演示 checkpoint
保存方式。仓库不提交这两个文件，因为模型权重体积较大，且可通过 Notebook 重新
生成。`artifacts/metrics.json` 中保留了运行时 checkpoint 路径，这不表示仓库包含
对应权重。

## 如何运行

仓库不附带 IMDb 数据和 GloVe 文件。准备以下路径：

```text
data/
  aclImdb/
    train/
      pos/
      neg/
  glove/
    glove.6B.300d.txt
```

也可以设置：

```powershell
$env:IMDB_TRAIN_DIR = "D:\datasets\aclImdb\train"
$env:GLOVE_PATH = "D:\datasets\glove.6B.300d.txt"
```

安装和启动：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

完整运行会训练两个模型，每个模型 10 个 epoch。当前执行环境为 Python 3.12.14、
PyTorch 2.14.1+cu126、CUDA 12.6 和 RTX 4060 Laptop GPU。
