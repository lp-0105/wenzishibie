# 文字识别项目 (OCR Project)

本项目使用机器学习/深度学习技术进行文字识别。包含完整的数据预处理、训练和预测脚本。

## 📁 目录结构

- [model.py](model.py): 模型架构定义。
- [train_scratch.py](train_scratch.py): 训练脚本。
- [prepare_data.py](prepare_data.py): 数据集预处理脚本（解压、划分、生成字典）。
- [predict.py](predict.py): 推理/预测脚本。
- [build_dict.py](build_dict.py): 构建字符字典。

## 🚀 快速开始

### 1. 准备数据
请先下载原始数据集并存放在 `data/` 目录下。然后运行以下命令进行数据预处理：

```bash
python prepare_data.py
```

该脚本将：
- 解压原始图片。
- 按照比例划分训练集与验证集。
- 在 `train_data/` 目录下生成 `train.txt`, `val.txt` 和 `dict.txt`。

### 2. 开始训练

```bash
python train_scratch.py
```

### 3. 模型预测

```bash
python predict.py
```

## 📝 实验日志
详细的开发过程和失败分析请参考：
- [实验报告_2026-04-01.md](实验报告_2026-04-01.md)
- [实验报告_2026-03-31_失败分析.md](实验报告_2026-03-31_失败分析.md)

## 📦 注意事项
- 巨大的 `train_data/` 文件夹和模型权重 `checkpoints/` 已通过 `.gitignore` 排除，不建议直接上传至 GitHub。
- 请根据自己的开发环境修改 [prepare_data.py](prepare_data.py) 中的路径配置。
