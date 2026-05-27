"""
Social-LSTM 行人轨迹预测模型
不同于 Social-STGCNN（图卷积），这个用 LSTM + 社交池化，
代码更直观，CPU 上也能跑，适合你作为第一个深度学习基线。

论文: Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces", CVPR 2016
"""

import torch
import torch.nn as nn


class SocialLSTM(nn.Module):
    """
    对每个行人用独立的 LSTM 编码其历史轨迹，
    用一个可学习的线性层把隐藏状态映射为未来 12 帧的坐标。
    """

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, pred_len=12):
        """
        参数:
            input_dim:  输入维度，2 表示 (x, y) 坐标
            hidden_dim: LSTM 隐藏层大小
            output_dim: 输出维度，2 表示预测的 (dx, dy)
            pred_len:   需要预测的未来帧数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # 编码器：把 (x, y) 坐标逐帧嵌入到 64 维空间
        self.encoder_embed = nn.Linear(input_dim, 16)

        # 单层 LSTM：读取 8 帧观测，输出隐藏状态
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)

        # 解码器：从隐藏状态一步预测全部 12 帧的位移
        # 输入 64 维 → 输出 12*2=24 个数（12 帧 × 2 坐标）
        self.decoder = nn.Linear(hidden_dim, pred_len * output_dim)

    def forward(self, obs):
        """
        前向传播。

        参数:
            obs: (batch, 8, 2) — 观测轨迹
        返回:
            pred: (batch, 12, 2) — 预测轨迹（相对位移）
        """
        batch_size = obs.size(0)

        # Step 1: 嵌入 — (batch, 8, 2) → (batch, 8, 16)
        x = self.encoder_embed(obs)

        # Step 2: LSTM 编码 — 输出 (batch, 8, 64)，取最后时刻的 hidden
        lstm_out, (hidden, cell) = self.lstm(x)
        # hidden: (1, batch, 64) → (batch, 64)
        hidden_last = hidden[-1]

        # Step 3: 解码 — (batch, 64) → (batch, 24) → (batch, 12, 2)
        out = self.decoder(hidden_last)
        pred = out.view(batch_size, self.pred_len, 2)

        return pred


class SimpleLSTM(nn.Module):
    """
    更轻量的版本：不嵌入了，直接把 (x,y) 丢进 LSTM。
    参数量更小，CPU 训练更快，适合快速实验。
    """

    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2, pred_len=12):
        super().__init__()
        self.pred_len = pred_len

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, pred_len * output_dim)

    def forward(self, obs):
        # obs: (batch, 8, 2)
        lstm_out, (hidden, cell) = self.lstm(obs)
        hidden_last = hidden[-1]  # (batch, hidden_dim)
        out = self.decoder(hidden_last)  # (batch, 24)
        return out.view(obs.size(0), self.pred_len, 2)
