"""
训练脚本 — Social-LSTM + 社交池化，支持场景批处理加速
"""

import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    OBS_LEN, PRED_LEN, LR, EPOCHS, CHECKPOINT_DIR,
    SOCIAL_EMBEDDING_DIM, SOCIAL_HIDDEN_DIM,
    SOCIAL_GRID_SIZE, SOCIAL_NEIGHBORHOOD_SIZE,
    SOCIAL_ACCUM_STEPS, SOCIAL_WINDOW_STRIDE, SOCIAL_GRAD_CLIP,
)
from src.model import SocialLSTM
from src.scene_data_loader import build_scene_dataset


# 场景批大小 — 每次拼 N 个场景一起前向，增加矩阵尺寸榨干 CPU
SCENE_BATCH = 6


def pack_scenes(windows):
    """把一组场景窗口打包成一个大张量"""
    total_peds = sum(len(w.ped_ids) for w in windows)
    obs = torch.zeros(total_peds, OBS_LEN, 2)
    preds = torch.zeros(total_peds, PRED_LEN, 2)
    scene_ids = torch.zeros(total_peds, dtype=torch.long)
    ped_counts = []

    offset = 0
    for sid, w in enumerate(windows):
        P = len(w.ped_ids)
        obs[offset:offset + P] = torch.from_numpy(w.obs_traj)
        preds[offset:offset + P] = torch.from_numpy(w.pred_traj)
        scene_ids[offset:offset + P] = sid
        ped_counts.append(P)
        offset += P

    return obs, preds, scene_ids, ped_counts


def train_epoch(model, windows, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_peds = 0
    random.shuffle(windows)

    optimizer.zero_grad()
    n = len(windows)

    for batch_start in range(0, n, SCENE_BATCH):
        batch_end = min(batch_start + SCENE_BATCH, n)
        batch_wins = windows[batch_start:batch_end]

        obs, preds, scene_ids, ped_counts = pack_scenes(batch_wins)
        obs = obs.to(device)
        preds = preds.to(device)
        scene_ids = scene_ids.to(device)

        output = model(obs, scene_ids)  # (total_P, 12, 2)

        # 按每个场景独立算 loss，再平均
        loss = 0.0
        offset = 0
        for P in ped_counts:
            loss += criterion(output[offset:offset + P], preds[offset:offset + P]) / P
            offset += P
        loss = loss / len(ped_counts)
        loss.backward()

        total_loss += loss.item() * sum(ped_counts)
        total_peds += sum(ped_counts)

        batch_idx = batch_start // SCENE_BATCH + 1
        if batch_idx % SOCIAL_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), SOCIAL_GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

    total_batches = (n + SCENE_BATCH - 1) // SCENE_BATCH
    if total_batches % SOCIAL_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), SOCIAL_GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_peds if total_peds > 0 else 0.0


@torch.no_grad()
def validate(model, windows, device):
    model.eval()
    ade_sum, fde_sum, count = 0.0, 0.0, 0

    for batch_start in range(0, len(windows), SCENE_BATCH):
        batch_end = min(batch_start + SCENE_BATCH, len(windows))
        batch_wins = windows[batch_start:batch_end]

        obs, preds, scene_ids, ped_counts = pack_scenes(batch_wins)
        obs = obs.to(device)
        preds = preds.to(device)
        scene_ids = scene_ids.to(device)

        output = model(obs, scene_ids)

        diff = output - preds
        ade = torch.norm(diff, dim=2).mean(dim=1)
        fde = torch.norm(diff[:, -1, :], dim=1)

        ade_sum += ade.sum().item()
        fde_sum += fde.sum().item()
        count += obs.size(0)

    return ade_sum / count, fde_sum / count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    social_stgcnn_data = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "Social-STGCNN", "datasets"
    )
    if not os.path.exists(social_stgcnn_data):
        print(f"数据目录不存在: {social_stgcnn_data}")
        print("请先克隆: git clone https://github.com/abduallahmohamed/Social-STGCNN.git")
        return

    print("构建场景数据集...")
    all_windows = build_scene_dataset(
        social_stgcnn_data,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        stride=SOCIAL_WINDOW_STRIDE,
    )
    print(f"场景窗口总数: {len(all_windows)}")

    if len(all_windows) == 0:
        print("错误: 没有提取到任何场景窗口")
        return

    ped_counts = [len(w.ped_ids) for w in all_windows]
    print(f"每场景行人数: min={min(ped_counts)}, max={max(ped_counts)}, "
          f"avg={np.mean(ped_counts):.1f}")

    random.seed(42)
    random.shuffle(all_windows)
    split = int(len(all_windows) * 0.8)
    train_windows = all_windows[:split]
    val_windows = all_windows[split:]
    print(f"训练场景: {len(train_windows)} | 验证场景: {len(val_windows)}")
    print(f"场景批大小: {SCENE_BATCH} | 预计每批 {np.mean(ped_counts) * SCENE_BATCH:.0f} 行人")

    model = SocialLSTM(
        embedding_dim=SOCIAL_EMBEDDING_DIM,
        hidden_dim=SOCIAL_HIDDEN_DIM,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        grid_size=SOCIAL_GRID_SIZE,
        neighborhood_size=SOCIAL_NEIGHBORHOOD_SIZE,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {param_count:,}")
    print(f"  embedding_dim={SOCIAL_EMBEDDING_DIM}, "
          f"hidden_dim={SOCIAL_HIDDEN_DIM}, "
          f"grid={SOCIAL_GRID_SIZE}x{SOCIAL_GRID_SIZE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()
    print(f"累积步数: {SOCIAL_ACCUM_STEPS} | 梯度裁剪: {SOCIAL_GRAD_CLIP}")
    print("=" * 60)

    best_ade = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_windows, optimizer, criterion, device)
        val_ade, val_fde = validate(model, val_windows, device)
        scheduler.step(val_ade)

        if val_ade < best_ade:
            best_ade = val_ade
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "best_model.pth"))

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss:.4f} | "
                  f"ADE: {val_ade:.3f}m | FDE: {val_fde:.3f}m | {elapsed:.0f}s")

    print("=" * 60)
    print(f"\n训练完成! 最佳验证 ADE: {best_ade:.3f}m")
    print(f"模型已保存到: {CHECKPOINT_DIR}/best_model.pth")
    print("对比基线: SimpleLSTM 约 0.37m | 匀速基线 0.48m")


if __name__ == "__main__":
    main()
