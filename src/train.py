"""
训练脚本 — 用 ETH/UCY 数据集训练 Social-LSTM 轨迹预测模型
纯 CPU 可跑（建议用 Colab GPU 加速），每 epoch 约 30 秒
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import OBS_LEN, PRED_LEN, BATCH_SIZE, LR, EPOCHS, CHECKPOINT_DIR
from src.data_loader import load_eth_ucy_file, extract_trajectories, normalize_trajectories
from src.model import SimpleLSTM, SocialLSTM


# ================================================================
# PyTorch Dataset — 把 numpy 数据包成 DataLoader 能吃的格式
# ================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, obs, preds):
        """
        obs: (N, 8, 2) numpy 数组 — 观测轨迹
        preds: (N, 12, 2) numpy 数组 — 真实未来轨迹
        """
        self.obs = torch.from_numpy(obs).float()
        self.preds = torch.from_numpy(preds).float()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.preds[idx]


# ================================================================
# 训练一个 epoch
# ================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for obs, preds in dataloader:
        obs = obs.to(device)        # (batch, 8, 2)
        preds = preds.to(device)    # (batch, 12, 2)

        optimizer.zero_grad()
        output = model(obs)         # (batch, 12, 2)
        loss = criterion(output, preds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * obs.size(0)

    return total_loss / len(dataloader.dataset)


# ================================================================
# 验证 — 算 ADE 和 FDE
# ================================================================
@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    ade_sum, fde_sum, count = 0.0, 0.0, 0
    for obs, preds in dataloader:
        obs = obs.to(device)
        preds = preds.to(device)
        output = model(obs)  # (batch, 12, 2)

        # ADE: 所有时间步的平均 L2 距离
        diff = output - preds  # (batch, 12, 2)
        ade = torch.norm(diff, dim=2).mean(dim=1)  # (batch,) — 每条轨迹的平均误差

        # FDE: 最后一帧的 L2 距离
        fde = torch.norm(diff[:, -1, :], dim=1)     # (batch,)

        ade_sum += ade.sum().item()
        fde_sum += fde.sum().item()
        count += obs.size(0)

    return ade_sum / count, fde_sum / count


# ================================================================
# 主流程
# ================================================================
def main():
    # ---- 检测设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cpu":
        print("提示: CPU 训练较慢，建议把代码上传 Colab 用免费 GPU")

    # ---- 加载数据 ----
    print("\n加载数据集...")
    social_stgcnn_data = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "Social-STGCNN", "datasets"
    )

    # 递归查找所有 .txt 文件，去重（同一个文件名只加载一次）
    seen = set()
    all_files = []
    for root, dirs, files in os.walk(social_stgcnn_data):
        # 跳过 raw 和 test-cpu 目录（原始未拆分数据）
        if "raw" in root or "test-cpu" in root:
            continue
        for f in files:
            if f.endswith(".txt") and f not in seen:
                seen.add(f)
                all_files.append(os.path.join(root, f))

    print(f"找到 {len(all_files)} 个唯一数据文件")
    datasets = {}
    for fpath in sorted(all_files):
        fname = os.path.basename(fpath)
        raw = load_eth_ucy_file(fpath)
        obs, preds = extract_trajectories(raw)
        if len(obs) == 0:
            continue
        print(f"  {os.path.basename(os.path.dirname(fpath))}/{fname}: {len(obs)} 条")

        # 文件名 → 数据集名映射
        fname_lower = fname.lower()
        for keyword, key in [("eth", "eth"), ("hotel", "hotel"),
                              ("zara01", "zara1"), ("zara02", "zara2"),
                              ("zara03", "zara3"), ("students", "univ"),
                              ("uni_examples", "univ")]:
            if keyword in fname_lower:
                dataset_key = key
                break
        else:
            dataset_key = "other"

        if dataset_key not in datasets:
            datasets[dataset_key] = {"train": ([], []), "val": ([], [])}

        if "val" in fname_lower:
            datasets[dataset_key]["val"][0].append(obs)
            datasets[dataset_key]["val"][1].append(preds)
        else:
            datasets[dataset_key]["train"][0].append(obs)
            datasets[dataset_key]["train"][1].append(preds)

    # 合并每个数据集
    for name in datasets:
        for split in ["train", "val"]:
            obs_list, preds_list = datasets[name][split]
            if obs_list:
                datasets[name][split] = (
                    np.concatenate(obs_list), np.concatenate(preds_list)
                )
            else:
                datasets[name][split] = (np.array([]), np.array([]))

    # 选第一个有数据的数据集来训练
    train_obs, train_preds = None, None
    val_obs, val_preds = None, None
    for name in ["eth", "hotel", "univ", "zara1", "zara2"]:
        t_obs, t_preds = datasets[name]["train"]
        v_obs, v_preds = datasets[name]["val"]
        if len(t_obs) > 0:
            train_obs, train_preds = t_obs, t_preds
            val_obs, val_preds = v_obs, v_preds
            print(f"\n使用数据集: {name}")
            break

    if train_obs is None:
        print("错误: 没有找到训练数据")
        return

    # ---- 归一化 ----
    train_obs_n, train_preds_n = normalize_trajectories(train_obs, train_preds)
    if len(val_obs) > 0:
        val_obs_n, val_preds_n = normalize_trajectories(val_obs, val_preds)
    else:
        # 没有验证集的话用训练集最后 20% 做验证
        split = int(len(train_obs_n) * 0.8)
        val_obs_n, val_preds_n = train_obs_n[split:], train_preds_n[split:]
        train_obs_n, train_preds_n = train_obs_n[:split], train_preds_n[:split]

    print(f"训练集: {len(train_obs_n)} 条 | 验证集: {len(val_obs_n)} 条")

    # ---- DataLoader ----
    train_dataset = TrajectoryDataset(train_obs_n, train_preds_n)
    val_dataset = TrajectoryDataset(val_obs_n, val_preds_n)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- 模型、损失函数、优化器 ----
    # 先用轻量版 SimpleLSTM，CPU 上也能跑
    model = SimpleLSTM(hidden_dim=32, pred_len=PRED_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # ---- 训练循环 ----
    best_ade = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_ade, val_fde = validate(model, val_loader, device)
        scheduler.step(val_ade)  # 根据验证 ADE 调整学习率

        elapsed = time.time() - t0

        # 保存最佳模型
        if val_ade < best_ade:
            best_ade = val_ade
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "best_model.pth"))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Loss: {train_loss:.4f} | "
                  f"ADE: {val_ade:.3f}m | FDE: {val_fde:.3f}m | "
                  f"Time: {elapsed:.1f}s")

    print("=" * 60)
    print(f"\n训练完成！最佳验证 ADE: {best_ade:.3f} 米")
    print(f"模型已保存到: {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
