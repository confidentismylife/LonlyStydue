"""Social Pooling 模块 — 4×4 空间网格，对邻居的隐藏状态求和池化"""

import torch
import torch.nn as nn


class SocialPooling(nn.Module):
    """
    4×4 空间网格池化层

    以目标行人 i 为中心，将边长 neighborhood_size 的正方形区域划分成
    grid_size × grid_size 的网格。邻居 j 按其相对 i 的位置落入对应网格单元，
    同一单元内所有邻居的隐藏状态求和，作为该单元的社交特征。

    scene_ids 参数用于批量场景训练：不同场景的行人不交互。
    """

    def __init__(self, grid_size=4, neighborhood_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.half_size = neighborhood_size / 2.0

    def forward(self, positions, hidden_states, scene_ids=None):
        """
        positions:      (P, 2)
        hidden_states:  (P, H)
        scene_ids:      (P,) optional — 每个行人所属场景的 ID

        Returns:
            pooled: (P, grid_size, grid_size, H)
        """
        P, H = hidden_states.shape
        device = positions.device

        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (P, P, 2)

        cell_idx = torch.floor(
            (rel_pos + self.half_size) / self.neighborhood_size * self.grid_size
        ).long()  # (P, P, 2)

        in_bounds = (
            (cell_idx[:, :, 0] >= 0) & (cell_idx[:, :, 0] < self.grid_size) &
            (cell_idx[:, :, 1] >= 0) & (cell_idx[:, :, 1] < self.grid_size)
        )
        not_self = ~torch.eye(P, dtype=torch.bool, device=device)

        valid_mask = in_bounds & not_self

        # 场景隔离：不同场景的行人不交互
        if scene_ids is not None:
            same_scene = scene_ids.unsqueeze(0) == scene_ids.unsqueeze(1)
            valid_mask = valid_mask & same_scene

        pooled = torch.zeros(P, self.grid_size, self.grid_size, H, device=device)

        for m in range(self.grid_size):
            for n in range(self.grid_size):
                cell_mask = valid_mask & (
                    (cell_idx[:, :, 0] == n) & (cell_idx[:, :, 1] == m)
                )
                pooled[:, m, n, :] = torch.matmul(
                    cell_mask.float(), hidden_states
                )

        return pooled
