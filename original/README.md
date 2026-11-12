# 原版 LSTM：可执行 Notebook 与当前重跑结果

## 目录定位

这里保留课程项目阶段的原始 LSTM 实验流程。它不是严格受控实验，而是用于展示
项目起点、代码演进和后来发现的问题。

主要负责内容：

- 从 IMDb `train/pos` 和 `train/neg` 读取 25,000 条评论
- 按 80% / 10% / 10% 划分为 20,000 / 2,500 / 2,500 条
- 使用 BeautifulSoup 删除 HTML 标签
- 转小写并用正则删除数字、标点和特殊字符
- 使用 NLTK 分词和英文停用词表
- 从训练集构建 10,000 词词表
- 将 token 转换为整数 ID，并在每个 batch 内 padding
- 定义单层单向 LSTM
- 训练 Random Embedding 和冻结 GloVe Embedding 两个模型
- 计算 Accuracy、Precision、Recall、F1、混淆矩阵和 ROC-AUC

## 文件

| 文件 | 内容 |
| --- | --- |
| `Lstm_model_train_and_test_original.ipynb` | 基于原课程代码的可执行 Notebook，保留一次真实完整运行输出 |
| `requirements.txt` | Notebook 所需 Python 依赖 |
| `configs/experiment.json` | 原版流程的数据、预处理、模型和训练配置 |
| `configs/resume_reference.json` | 简历中保留的历史指标 |
| `configs/executed_environment.json` | 本次真实运行使用的软件和硬件环境 |
| `artifacts/metrics.json` | 本次运行的完整指标和逐轮 Loss |
| `artifacts/loss_curves.png` | 两组模型的训练与验证 Loss 曲线 |
| `artifacts/metrics_comparison.png` | Accuracy、F1 和 ROC-AUC 对比 |

## 原版流程

```text
IMDb train/pos 和 train/neg
  -> 80/10/10 随机划分
  -> HTML 清理、小写化和正则清洗
  -> NLTK 分词并删除默认停用词
  -> 仅用训练集构建 10,000 词词表
  -> token 转整数 ID
  -> batch 内 padding
  -> Embedding
  -> 单层单向 LSTM
  -> 最后时间步 hidden state
  -> Dropout
  -> Linear
  -> CrossEntropyLoss
```

Random Embedding 保持可训练；GloVe Embedding 使用 `glove.6B.300d.txt` 初始化后
冻结。padding 之后的 `<pad>` ID 仍然作为 LSTM 时间步输入。

## 历史简历结果

| 模型 | Accuracy | ROC-AUC | 说明 |
| --- | ---: | ---: | --- |
| LSTM + Random Embedding | 80.76% | 0.8630 | 历史实验 |
| LSTM + frozen GloVe | 73.52% | 0.7730 | 历史实验 |

历史记录中 Random 的 Accuracy 比 GloVe 高 7.24 个百分点。这个结果已经写入
简历，本仓库将它作为历史记录保留。

## 当前环境重跑结果

为了让提交的 Notebook 可以在当前电脑上稳定复核，本次运行在原流程上增加了
seed 42、CUDA 确定性设置和带 generator 的 DataLoader。除此之外，没有修正
Embedding 可训练状态、padding 和最后一轮测试等实验设计问题。

本次真实结果：

| Embedding | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random，可训练 | 0.8624 | 0.8702 | 0.8550 | 0.8625 | 0.9238 |
| GloVe，冻结 | 0.8192 | 0.8115 | 0.8360 | 0.8236 | 0.8903 |

Random 的 Accuracy 高 4.32 个百分点，ROC-AUC 高 0.0335。排名方向与历史记录
一致，但绝对指标和差值没有精确复现。当前数值应表述为：

> 在当前软件、硬件和固定执行路径下重跑原版代码，Random Embedding 的结果仍
> 高于冻结 GloVe，但差距由历史记录的 7.24 个百分点缩小到 4.32 个百分点。

不能表述为“完全复现了简历中的 80.76% / 73.52%”。环境、随机初始化、CUDA
算子、数据划分顺序和代码修复都可能影响最终数值。

## 原版仍然存在的主要问题

### 1. Embedding 公平性

Random Embedding 会继续训练，GloVe 被冻结。两组同时改变了初始化来源和是否
可训练两个因素，因此不是严格单变量比较。

### 2. Padding 进入 LSTM

同一 batch 的短评论被补成最大长度，但模型没有接收真实 lengths。padding 会
继续经过 LSTM 的多个时间步，并可能改变最终 hidden state。

### 3. 使用最后一轮模型

代码训练 10 个 epoch 后直接测试最后状态，没有根据验证集保存最佳 checkpoint。
如果训练后期过拟合，最后一轮不一定是验证表现最好的模型。

### 4. 测试集不是官方 IMDb 测试集

当前 2,500 条测试评论来自 IMDb 的 25,000 条训练数据。它不是 IMDb 官方
`test/pos` 和 `test/neg` 的 25,000 条测试集，因此结果不能与其他使用官方测试集
的报告直接比较。

### 5. 只运行一个种子

即使本次增加了固定 seed，单次结果仍不能描述初始化、Dropout 和数据顺序带来的
波动，也无法计算均值和标准差。

### 6. 词表大小没有消融

10,000 是人工设定的折中值，没有在验证集比较 5,000、10,000、20,000 等候选，
不能称为已经验证的最优大小。

## 如何运行

仓库不附带 IMDb 数据和 GloVe 文件。准备以下目录：

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

安装依赖并启动 Jupyter：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

完整训练会运行两个模型，每个 10 个 epoch。模型权重不会提交到仓库。
