#!/bin/bash
# QAT 项目依赖安装脚本
# 适用于 CUDA 12.x 环境

# 安装 PyTorch 2.1.0 with CUDA 12.1 (兼容 CUDA 12.8)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 PyTorch Geometric 相关包 (兼容 PyTorch 2.1.0 + CUDA 12.1)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric

# 安装 transformers (使用较新的版本，但保持向后兼容)
pip install transformers==4.30.0

# 其他依赖包
pip install numpy==1.24.4
pip install spacy==3.5.3
pip install nltk sentencepiece tqdm tensorboard
pip install einops timm wandb torchtext==0.16.0

# 下载 spacy 英语模型（如果网络连接失败可以跳过）
python -m spacy download en_core_web_sm || echo "Warning: spacy 模型下载失败，可以稍后手动下载"

echo "✓ 依赖安装完成！"
