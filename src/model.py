"""
Social-LSTM 行人轨迹预测模型
论文: Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces", CVPR 2016
"""

import torch
import torch.nn as nn

from src.social_pooling import SocialPooling


class SocialLSTM(nn.Module):
    """
    完整 Social-LSTM: 用 LSTMCell 逐时间步处理，每步对周围行人的隐藏状态
    做社交池化，让每个行人"看见"周围的人。

    编码阶段 (t=0..obs_len-1): 逐帧输入观测位置 → 社交池化 → LSTMCell
    解码阶段 (t=obs_len..obs_len+pred_len-1): 自回归预测位移，预测位置继续社交池化
    """

    def __init__(self, embedding_dim=64, hidden_dim=128, output_dim=2,
                 obs_len=8, pred_len=12, grid_size=4, neighborhood_size=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.pos_embedding = nn.Linear(2, embedding_dim)
        self.social_pool = SocialPooling(grid_size=grid_size,
                                         neighborhood_size=neighborhood_size)
        self.social_embedding = nn.Linear(
            grid_size * grid_size * hidden_dim, embedding_dim
        )
        self.lstm_cell = nn.LSTMCell(
            input_size=embedding_dim * 2, hidden_size=hidden_dim
        )
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, obs, scene_ids=None):
        """
        obs:       (P, obs_len, 2) — P 个行人的观测轨迹（已归一化）
        scene_ids: (P,) optional — 场景 ID，批量训练时隔离不同场景

        Returns:
            pred: (P, pred_len, 2) — 预测位移
        """
        P = obs.size(0)
        H = self.hidden_dim
        device = obs.device

        h_t = torch.zeros(P, H, device=device)
        c_t = torch.zeros(P, H, device=device)

        for t in range(self.obs_len):
            pos_t = obs[:, t, :]
            pos_emb = self.relu(self.pos_embedding(pos_t))
            pooled = self.social_pool(pos_t, h_t, scene_ids)
            pooled_flat = pooled.view(P, -1)
            social_emb = self.relu(self.social_embedding(pooled_flat))
            lstm_input = torch.cat([pos_emb, social_emb], dim=1)
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))

        current_pos = obs[:, -1, :]
        displacements = []

        for t in range(self.pred_len):
            pos_emb = self.relu(self.pos_embedding(current_pos))
            pooled = self.social_pool(current_pos, h_t, scene_ids)
            pooled_flat = pooled.view(P, -1)
            social_emb = self.relu(self.social_embedding(pooled_flat))
            lstm_input = torch.cat([pos_emb, social_emb], dim=1)
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))
            displacement = self.output_layer(h_t)
            displacements.append(displacement)
            current_pos = current_pos + displacement

        return torch.stack(displacements, dim=1)


class SimpleLSTM(nn.Module):
    """
    轻量基线版本：不嵌入了，直接把 (x,y) 丢进 LSTM。
    每条轨迹独立预测，无社交交互。
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
        out = self.decoder(hidden_last)
        return out.view(obs.size(0), self.pred_len, 2)
