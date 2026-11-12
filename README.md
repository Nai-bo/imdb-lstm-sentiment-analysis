# IMDb 电影评论情感分析：LSTM 原版与改进版

本项目整理本人完成的 LSTM 实验，包括：

- IMDb 评论读取与训练集、验证集、测试集划分
- HTML 清理、小写化、正则清洗、停用词处理与分词
- 基于训练集构建词表并将 token 转换为整数 ID
- PyTorch LSTM 情感分类模型定义
- 训练、验证、最佳 checkpoint 选择与测试评估
- Random Embedding 与 GloVe Embedding 的对照实验
- Accuracy、Precision、Recall、F1、混淆矩阵和 ROC-AUC 分析

## 先看什么

| 目标 | 文件 |
| --- | --- |
| 查看原版可执行 Notebook 和真实运行输出 | [`original/Lstm_model_train_and_test_original.ipynb`](original/Lstm_model_train_and_test_original.ipynb) |
| 了解原版保留的问题与当前重跑结果 | [`original/README.md`](original/README.md) |
| 查看小范围改进后的 Notebook | [`improved/Lstm_model_train_and_test_improved.ipynb`](improved/Lstm_model_train_and_test_improved.ipynb) |
| 查看改进设计、结果和限制 | [`improved/README.md`](improved/README.md) |

两个 Notebook 都已完整执行，运行结果保留在对应代码单元下方。导入、函数定义和
类定义单元没有自然输出，因此这些单元下方为空是正常现象。

## 两版实验的关系

改进版以原 Notebook 为基础，没有替换成另一套项目框架。数据读取、80% / 10% /
10% 划分、清洗方式、10,000 词词表、单层单向 LSTM 和主要训练配置保持一致。

| 项目 | 原版 | 改进版 |
| --- | --- | --- |
| 定位 | 保留课程项目流程，并让当前环境可以完整运行 | 在不改变主体结构的前提下修正主要实验问题 |
| 数据量 | 20,000 / 2,500 / 2,500 | 20,000 / 2,500 / 2,500 |
| 测试集 | 从 IMDb 的 25,000 条训练评论中重新划分 | 与原版相同，不是 IMDb 官方 25,000 条测试集 |
| 随机性 | 历史实验未完整固定；当前可运行版增加 seed 42 和 DataLoader generator | 固定 Python、NumPy、PyTorch、CUDA 和 DataLoader 随机性 |
| Embedding 对照 | Random 可训练，GloVe 冻结 | 两者都冻结 |
| Padding | `<pad>` 会进入 LSTM 时间步 | 使用 `pack_padded_sequence` 排除 padding |
| 模型选择 | 测试最后一轮 | 按验证集 Accuracy 选择最佳轮次，平手时看验证 Loss |
| 指标 | 手写计算 | 使用 scikit-learn 自动计算 |
| 输出 | Loss 曲线和指标比较图 | Loss 曲线、指标比较图、ROC 曲线和指标 JSON |

原版当前可运行版本增加了统一随机种子和 DataLoader generator，这是为了让仓库中
保留的结果可以复核。除此之外，原版仍有 Random 可训练、GloVe 冻结、padding
进入 LSTM 和使用最后一轮模型等设计问题。

## 结果

### 1. 历史简历记录

以下数字来自课程项目当时提交到简历中的记录：

| 模型 | Accuracy | ROC-AUC | 说明 |
| --- | ---: | ---: | --- |
| LSTM + Random Embedding | 80.76% | 0.8630 | 历史实验 |
| LSTM + frozen GloVe | 73.52% | 0.7730 | 历史实验 |

历史记录中 Random Embedding 的 Accuracy 比 GloVe 高 7.24 个百分点。但由于两组
实验的 Embedding 可训练状态不同，且随机性、checkpoint 选择和 padding 处理没有
完整控制，这个差值不能作为严格的因果结论。

### 2. 原版当前重跑

当前环境使用 seed 42 重跑原版流程，结果如下：

| Embedding | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random，可训练 | 0.8624 | 0.8702 | 0.8550 | 0.8625 | 0.9238 |
| GloVe，冻结 | 0.8192 | 0.8115 | 0.8360 | 0.8236 | 0.8903 |

Random 的 Accuracy 仍然更高，方向与历史记录一致，但差距缩小为 4.32 个百分点；
ROC-AUC 的差距为 0.0335。当前结果没有精确复现简历中的 80.76% / 73.52% 和
7.24 个百分点差距，不能把两种情况描述成同一次数值复现。

### 3. 改进版当前运行

改进版仍使用 seed 42 和原数据划分，只运行一次。结果如下：

| Embedding | Accuracy | Precision | Recall | F1 | ROC-AUC | 最佳轮次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random，冻结 | 0.8340 | 0.8413 | 0.8273 | 0.8342 | 0.9093 | 7 |
| GloVe，冻结 | 0.8632 | 0.8840 | 0.8391 | 0.8610 | 0.9376 | 7 |

在这组受控重跑中，GloVe 的 Accuracy 比 Random 高 2.92 个百分点，ROC-AUC 高
0.0283。方向与历史记录相反。

这组结果说明原版的 7.24 个百分点排名并不稳定，但不能反过来证明 GloVe 在所有
场景中一定更好。当前只使用一个种子、一种 LSTM 结构和原项目自定义的 2,500 条
测试集；这些结果只适用于当前实验配置。

## 改进版修正了什么

| 原版问题 | 改进方式 | 目的 |
| --- | --- | --- |
| 初始化、Dropout 和 DataLoader 顺序存在随机波动 | 固定 Python、NumPy、PyTorch、CUDA 和 DataLoader seed | 让实际执行路径可以复核 |
| 两组 Embedding 的可训练状态不同 | Random 和 GloVe 都冻结 | 把比较集中在初始化来源上 |
| Padding 被当作真实 token 输入 LSTM | 记录真实 lengths 并使用 `pack_padded_sequence` | 避免补零改变最终 hidden state |
| 直接测试最后一轮 | 按验证集 Accuracy 保存最佳模型状态 | 测试集不参与模型选择 |
| 手写 Accuracy、F1 和 AUC | 使用 scikit-learn 计算 | 减少指标实现错误 |
| 只有终端文本结果 | 保存 Loss、指标图、ROC 和 JSON | 让过程和结果可检查 |

改进版没有加入官方 IMDb 测试集、早停、三随机种子重复实验或新的项目包。这样做的
目的是保持它与原始课程项目之间的可追溯关系。

## 当前问题与解释边界

- 只运行了 seed 42，没有报告多个种子的均值和标准差。
- 所谓测试集是原 25,000 条训练评论中的 2,500 条，不是 IMDb 官方测试集。
- 两组 Embedding 都冻结是一种受控比较，但不是寻找最高精度模型的完整方案。
- 没有对 5,000、10,000、20,000 等词表大小做消融实验。
- 没有单独调分类阈值、LSTM 层数、hidden size 或 Dropout。
- 没有对测试集做系统性错误分析，也没有统计显著性检验。
- 历史指标、原版当前重跑和改进版结果来自不同的实验控制条件，不能直接混用。

## 目录

```text
LSTM_Sentiment_Project/
  README.md
  .gitignore
  original/
    README.md
    Lstm_model_train_and_test_original.ipynb
    requirements.txt
    configs/
      experiment.json
      resume_reference.json
      executed_environment.json
    artifacts/
      metrics.json
      loss_curves.png
      metrics_comparison.png
  improved/
    README.md
    Lstm_model_train_and_test_improved.ipynb
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

## 如何运行

仓库不附带 IMDb 原始评论、GloVe 文件和模型权重，因为数据与权重体积较大。
Notebook 的中文说明、代码和已保存输出可以直接浏览；完整重跑需要准备本地数据。

建议使用 Python 3.10 至 3.12。PyTorch 应按照自己的 CPU/CUDA 环境安装。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r original\requirements.txt
jupyter lab
```

启动 Jupyter 后设置实际路径，或提前设置环境变量：

```powershell
$env:IMDB_TRAIN_DIR = "D:\datasets\aclImdb\train"
$env:GLOVE_PATH = "D:\datasets\glove.6B.300d.txt"
```

Notebook 会优先使用环境变量，其次从当前目录和父目录寻找以下相对路径：

```text
data/
  aclImdb/
    train/
      pos/
      neg/
  glove/
    glove.6B.300d.txt
```

首次使用 NLTK 时需要本地 `stopwords` 和 `punkt` 资源。Notebook 会先检查本地资源，
缺失时才尝试下载。

完整运行会训练两个模型，每个模型 10 个 epoch。当前测试环境为 Python 3.12.14、
PyTorch 2.14.1+cu126、CUDA 12.6 和 RTX 4060 Laptop GPU。不同 CUDA、PyTorch
或 cuDNN 版本可能造成小幅数值变化。

## 完成边界

本人完成数据划分、数据清理与分词、LSTM 模型构建、训练评估、随机 Embedding
与 GloVe 对照实验。
