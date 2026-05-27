"""
标准评估脚本 — 用训练时没见过的测试集评估模型
三种评估方式:
  1. 同场景测试: 用 Social-STGCNN 的 test 分集
  2. 跨场景测试: 在 A 场景训练，去 B 场景测试（论文标准做法）
  3. 全部测试集汇总: 把所有 test 文件合并评估
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections import defaultdict

from src.model import SimpleLSTM
from src.config import CHECKPOINT_DIR, OBS_LEN, PRED_LEN
from src.data_loader import load_eth_ucy_file, extract_trajectories, normalize_trajectories


def load_all_files(base_dir, split_filter="test"):
    """加载指定 split 的全部文件，按场景名分组"""
    datasets = defaultdict(lambda: ([], []))
    for root, dirs, files in os.walk(base_dir):
        if "raw" in root or "test-cpu" in root:
            continue
        for f in files:
            if not f.endswith(".txt"):
                continue
            fpath = os.path.join(root, f)
            fname = f.lower()

            # 只取指定的 split
            if split_filter not in os.path.basename(root):
                continue

            # 识别场景名
            for kw, scene in [("eth", "eth"), ("hotel", "hotel"),
                              ("zara01", "zara1"), ("zara02", "zara2"),
                              ("zara03", "zara3"), ("students", "univ"),
                              ("uni_examples", "univ")]:
                if kw in fname:
                    key = scene
                    break
            else:
                key = "other"

            raw = load_eth_ucy_file(fpath)
            obs, preds = extract_trajectories(raw, obs_len=OBS_LEN, pred_len=PRED_LEN)
            if len(obs) > 0:
                datasets[key][0].append(obs)
                datasets[key][1].append(preds)

    # 合并每个场景内的文件
    result = {}
    for scene, (obs_list, preds_list) in datasets.items():
        if obs_list:
            result[scene] = (
                np.concatenate(obs_list),
                np.concatenate(preds_list)
            )
    return result


@torch.no_grad()
def compute_metrics(model, obs, preds, device):
    """计算 ADE 和 FDE"""
    # 归一化
    origin = obs[:, -1:, :]
    obs_n = obs - origin
    preds_n = preds - origin

    # 推理
    x = torch.from_numpy(obs_n).float().to(device)
    output = model(x).cpu().numpy()  # (N, 12, 2)

    # 指标
    diff = output - preds_n  # (N, 12, 2)
    ade = np.mean(np.linalg.norm(diff, axis=2))       # 全部帧平均
    fde = np.mean(np.linalg.norm(diff[:, -1, :], axis=1))  # 最后一帧
    return ade, fde


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 加载模型 ----
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"未找到模型: {model_path}")
        print("请先在 Colab 上训练或运行 python src/train.py")
        return

    # 自动检测 checkpoint 的 hidden_dim
    ckpt = torch.load(model_path, map_location=device)
    # lstm.weight_ih_l0 shape = [4*hidden, input] → hidden = shape[0]//4
    hidden_dim = ckpt["lstm.weight_ih_l0"].shape[0] // 4
    model = SimpleLSTM(hidden_dim=hidden_dim, pred_len=PRED_LEN).to(device)
    model.load_state_dict(ckpt)
    del ckpt  # 释放内存
    model.eval()
    print(f"模型已加载: {model_path}\n")

    # ---- 数据目录 ----
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "..", "Social-STGCNN", "datasets")
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return

    # ================================================================
    # 评估 1: 同场景测试集
    # ================================================================
    print("=" * 65)
    print("  评估 1 — 各场景 test 分集 (模型在全部数据训练，测试集独立)")
    print("=" * 65)
    test_sets = load_all_files(data_dir, split_filter="test")

    all_ade, all_fde = [], []
    for scene in ["eth", "hotel", "univ", "zara1", "zara2"]:
        if scene not in test_sets:
            continue
        obs, preds = test_sets[scene]
        ade, fde = compute_metrics(model, obs, preds, device)
        all_ade.append(ade)
        all_fde.append(fde)
        print(f"  {scene:6s} | {len(obs):5d} 条 | ADE: {ade:.3f}m | FDE: {fde:.3f}m")

    if all_ade:
        print(f"  {'---':6s} | {'':5s} | {'─'*25}")
        print(f"  {'平均':6s} | {'':5s} | ADE: {np.mean(all_ade):.3f}m | "
              f"FDE: {np.mean(all_fde):.3f}m")

    # ================================================================
    # 评估 2: 跨场景泛化（论文标准）
    # ================================================================
    print("\n" + "=" * 65)
    print("  评估 2 — 跨场景泛化 (Leave-One-Out)")
    print("  在 4 个场景训练，去第 5 个场景测试")
    print("=" * 65)

    # 收集所有场景的 train 数据
    train_sets = load_all_files(data_dir, split_filter="train")

    # 合并除目标场景外的所有训练数据
    all_scenes = [s for s in train_sets.keys() if s in test_sets]
    for target in ["eth", "hotel", "univ", "zara1", "zara2"]:
        if target not in train_sets or target not in test_sets:
            continue

        # 用其他 4 个场景训练一个临时模型
        train_obs_list = []
        train_preds_list = []
        for s in all_scenes:
            if s != target:
                train_obs_list.append(train_sets[s][0])
                train_preds_list.append(train_sets[s][1])

        train_obs_all = np.concatenate(train_obs_list)
        train_preds_all = np.concatenate(train_preds_list)

        # 归一化
        origin = train_obs_all[:, -1:, :]
        train_obs_n = train_obs_all - origin
        train_preds_n = train_preds_all - origin

        # 快速训练（20 轮即可，跨场景评估不需要完整训练）
        quick_model = SimpleLSTM(hidden_dim=64, pred_len=PRED_LEN).to(device)
        opt = torch.optim.Adam(quick_model.parameters(), lr=0.001)
        crit = torch.nn.MSELoss()

        x_train = torch.from_numpy(train_obs_n).float().to(device)
        y_train = torch.from_numpy(train_preds_n).float().to(device)

        for ep in range(20):
            quick_model.train()
            opt.zero_grad()
            loss = crit(quick_model(x_train), y_train)
            loss.backward()
            opt.step()

            if ep == 19:  # 最后一轮打印
                quick_model.eval()
                test_obs, test_preds = test_sets[target]
                ade, fde = compute_metrics(quick_model, test_obs, test_preds, device)
                print(f"  训练: 4场景 → 测试: {target:6s} | "
                      f"Trains: {len(train_obs_all):5d} | Test: {len(test_obs):4d}条 | "
                      f"ADE: {ade:.3f}m | FDE: {fde:.3f}m")

    # ================================================================
    # 评估 3: 看几条预测 vs 真值
    # ================================================================
    print("\n" + "=" * 65)
    print("  评估 3 — 可视化 5 条测试集预测")
    print("=" * 65)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 找第一个有数据的 test 场景
    test_obs, test_preds = None, None
    for scene in ["eth", "hotel", "univ", "zara1", "zara2"]:
        if scene in test_sets:
            test_obs, test_preds = test_sets[scene]
            break

    if test_obs is not None:
        # 归一化 → 预测
        origin = test_obs[:, -1:, :]
        test_obs_n = test_obs - origin
        x = torch.from_numpy(test_obs_n).float().to(device)
        with torch.no_grad():
            pred_output = model(x).cpu().numpy()

        np.random.seed(42)
        ids = np.random.choice(len(test_obs), min(5, len(test_obs)), replace=False)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, idx in enumerate(ids):
            ax = axes[i]
            ax.plot(test_obs[idx, :, 0], test_obs[idx, :, 1],
                    'b-o', linewidth=2, markersize=4, label='Observed')
            ax.plot(pred_output[idx, :, 0], pred_output[idx, :, 1],
                    'r--o', linewidth=2, markersize=4, label='Predicted')
            ax.plot(test_preds[idx, :, 0], test_preds[idx, :, 1],
                    'g-o', linewidth=2, markersize=3, alpha=0.5, label='Ground Truth')
            ax.set_title(f'Test Sample {i+1}')
            ax.set_aspect('equal')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "test_results.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"图片已保存: {save_path}")
        print("蓝色=观测 | 红色=预测 | 绿色=真值")

    print("\n" + "=" * 65)
    print("  评估完成")
    print("  ADE越小越好 | FDE越小越好 | 和之前的基线 0.48m 对比")
    print("=" * 65)


if __name__ == "__main__":
    main()
