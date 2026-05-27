"""全局配置文件 — 所有路径和超参数集中管理"""

import os

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")

# ETH/UCY 数据集路径
ETH_PATH = os.path.join(DATA_RAW, "eth")
UCY_PATH = os.path.join(DATA_RAW, "ucy")

# 模型检查点
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

# ===== 训练参数（CPU友好） =====
BATCH_SIZE = 32
OBS_LEN = 8      # 观察 8 帧（3.2 秒）
PRED_LEN = 12    # 预测 12 帧（4.8 秒）
HIDDEN_DIM = 64
LR = 0.001
EPOCHS = 100

# 数据集 URL
ETH_UCY_URL = "https://github.com/crowdbotp/OpenTraj/raw/master/datasets/ETH_UCY/ethucy.zip"
