"""加载训练好的模型，对单条轨迹做预测并可视化"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.model import SimpleLSTM
from src.data_loader import load_eth_ucy_file, extract_trajectories, normalize_trajectories
from src.config import CHECKPOINT_DIR, OBS_LEN, PRED_LEN

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def predict_one(model, obs, device="cpu"):
    """输入一条 (8,2) 观测轨迹，返回 (12,2) 预测"""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # (1, 8, 2)
        pred = model(x)  # (1, 12, 2)
        return pred.squeeze(0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"未找到模型: {model_path}")
        print("请先运行: python src/train.py")
        return

    model = SimpleLSTM(hidden_dim=32, pred_len=PRED_LEN).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"模型已加载: {model_path}")

    # 找一条验证集数据做演示
    social_data = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "Social-STGCNN", "datasets", "eth", "val"
    )
    if not os.path.exists(social_data):
        print("未找到验证数据")
        return

    for fname in os.listdir(social_data):
        fpath = os.path.join(social_data, fname)
        raw = load_eth_ucy_file(fpath)
        obs_list, preds_list = extract_trajectories(raw)
        if len(obs_list) > 0:
            break

    # 归一化 → 预测 → 反归一化
    idx = np.random.randint(len(obs_list))
    obs, gt = obs_list[idx], preds_list[idx]
    origin = obs[-1:, :]
    obs_norm = obs - origin
    pred_norm = predict_one(model, obs_norm, device)
    pred = pred_norm + origin  # 回到绝对坐标系

    # 画图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(obs[:, 0], obs[:, 1], "b-o", linewidth=2, markersize=5, label="Observations (3.2s)")
    ax.scatter(obs[0, 0], obs[0, 1], c="green", s=80, zorder=5, label="Start")
    ax.plot(pred[:, 0], pred[:, 1], "r--o", linewidth=2, markersize=5, label="LSTM Prediction")
    ax.plot(gt[:, 0], gt[:, 1], "g-o", linewidth=2, markersize=3, alpha=0.5, label="Ground Truth")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Social-LSTM Trajectory Prediction")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "inference_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"推理结果已保存: {save_path}")


if __name__ == "__main__":
    main()
