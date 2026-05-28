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

# ===== Social-LSTM 参数 =====
SOCIAL_EMBEDDING_DIM = 64
SOCIAL_HIDDEN_DIM = 128
SOCIAL_GRID_SIZE = 4
SOCIAL_NEIGHBORHOOD_SIZE = 32
SOCIAL_ACCUM_STEPS = 1       # 梯度累积步数（每 N 个场景更新一次）
SOCIAL_WINDOW_STRIDE = 1     # 窗口采样步长（1=全量数据）
SOCIAL_CACHE = os.path.join(ROOT, "checkpoints", "scene_cache.pt")  # 场景缓存路径
SOCIAL_GRAD_CLIP = 1.0       # 梯度裁剪阈值

# 数据集 URL
ETH_UCY_URL = "https://github.com/crowdbotp/OpenTraj/raw/master/datasets/ETH_UCY/ethucy.zip"
