"""ETH/UCY 行人轨迹数据加载器 — 纯 CPU 运行"""

import os
import numpy as np
from collections import defaultdict
from src.config import OBS_LEN, PRED_LEN


def load_eth_ucy_file(filepath):
    """
    加载单个 ETH/UCY 文件
    格式: frame_id, ped_id, x, y  (有些有 z, 忽略)
    """
    data = defaultdict(list)  # {ped_id: [(frame, x, y), ...]}

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            frame = int(float(parts[0]))
            ped_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            data[ped_id].append((frame, x, y))

    return data


def extract_trajectories(data, obs_len=OBS_LEN, pred_len=PRED_LEN):
    """
    从原始数据中切分出 (观测, 预测) 轨迹对
    返回: obs_trajs (N, obs_len, 2), pred_trajs (N, pred_len, 2)
    """
    obs_list, pred_list = [], []

    for ped_id, traj in data.items():
        traj = sorted(traj, key=lambda t: t[0])  # 按帧排序
        seq_len = obs_len + pred_len

        # 滑动窗口切分
        for i in range(len(traj) - seq_len + 1):
            segment = traj[i : i + seq_len]
            obs = np.array([(t[1], t[2]) for t in segment[:obs_len]], dtype=np.float32)
            pred = np.array([(t[1], t[2]) for t in segment[obs_len:]], dtype=np.float32)
            obs_list.append(obs)
            pred_list.append(pred)

    if len(obs_list) == 0:
        return np.array([]), np.array([])
    return np.stack(obs_list), np.stack(pred_list)


def load_dataset(dataset_dir, dataset_name):
    """
    加载整个数据集（ETH 或 UCY 下的所有文件）
    返回汇总的 obs 和 pred
    """
    all_obs, all_preds = [], []

    for fname in sorted(os.listdir(dataset_dir)):
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.isfile(fpath):
            continue
        raw = load_eth_ucy_file(fpath)
        obs, preds = extract_trajectories(raw)
        if len(obs) > 0:
            all_obs.append(obs)
            all_preds.append(preds)
            print(f"  [{dataset_name}] {fname}: {len(obs)} 条轨迹")

    if len(all_obs) == 0:
        raise ValueError(f"数据集 {dataset_dir} 没有找到有效轨迹")
    return np.concatenate(all_obs), np.concatenate(all_preds)


def normalize_trajectories(obs, preds):
    """
    归一化：每条轨迹减去自身最后观测点的位置
    这是 Social-LSTM 论文的标准做法
    """
    # 以观测序列最后一帧的位置为原点
    origin = obs[:, -1:, :]  # (N, 1, 2)
    obs_norm = obs - origin
    preds_norm = preds - origin
    return obs_norm, preds_norm


def create_dataloader(obs, preds, batch_size=32, shuffle=True):
    """最简单的 numpy dataloader，不依赖 torch DataLoader"""
    idx = np.arange(len(obs))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i : i + batch_size]
        yield obs[batch_idx], preds[batch_idx]


if __name__ == "__main__":
    # 快速测试：打印数据集信息
    from src.config import ETH_PATH, UCY_PATH

    print("=" * 50)
    print("数据集加载测试")
    print("=" * 50)

    for name, path in [("ETH", ETH_PATH), ("UCY", UCY_PATH)]:
        print(f"\n--- {name} ---")
        if os.path.exists(path):
            obs, preds = load_dataset(path, name)
            print(f"总计: {len(obs)} 条轨迹, obs shape={obs.shape}, pred shape={preds.shape}")
        else:
            print(f"数据目录不存在: {path}")
            print("请先运行 src/download_data.py 下载数据")
