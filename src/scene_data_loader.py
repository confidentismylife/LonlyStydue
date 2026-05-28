"""
场景级数据加载器 — 按帧分组，提取共存行人的轨迹窗口

与 data_loader.py 的区别:
  data_loader.py: 按行人独立滑动窗口，丢失帧级场景上下文
  scene_data_loader.py: 按帧分组，同一时间步的所有行人组成一个场景
"""

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class SceneWindow:
    """一个场景窗口 — 同一时间窗口内共存的所有行人"""
    ped_ids: list          # 行人 ID 列表
    obs_traj: np.ndarray   # (P, obs_len, 2) 观测轨迹
    pred_traj: np.ndarray  # (P, pred_len, 2) 真实未来轨迹
    start_frame: float     # 起始帧号


def load_scene_data(filepath):
    """
    读取 ETH/UCY 格式文件，按帧分组

    Returns:
        frames: dict[frame_id -> dict[ped_id -> (x, y)]]
        ped_frames: dict[ped_id -> set of frame_ids] 每个行人出现的帧集合
    """
    frames = defaultdict(dict)
    ped_frames = defaultdict(set)

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().replace('\t', ' ').split()
            if len(parts) < 4:
                continue
            try:
                frame = float(parts[0])
                ped_id = int(float(parts[1]))
                x, y = float(parts[2]), float(parts[3])
                frames[frame][ped_id] = (x, y)
                ped_frames[ped_id].add(frame)
            except (ValueError, IndexError):
                continue

    return dict(frames), dict(ped_frames)


def _detect_frame_step(sorted_frames):
    """自动检测帧间隔（Social-STGCNN 数据统一为 10.0）"""
    if len(sorted_frames) < 2:
        return 10.0
    # 取最小的非零差值
    diffs = []
    for i in range(min(20, len(sorted_frames) - 1)):
        d = sorted_frames[i + 1] - sorted_frames[i]
        if d > 0:
            diffs.append(d)
    return min(diffs) if diffs else 10.0


def extract_scene_windows(frames, ped_frames, obs_len=8, pred_len=12):
    """
    从帧级数据中提取场景窗口

    对每个可能的起始帧，找到在该窗口的所有 20 帧中都有数据的行人，
    形成 (P, obs_len, 2) + (P, pred_len, 2) 的场景窗口

    Returns:
        list of SceneWindow
    """
    sorted_frames = sorted(frames.keys())
    if not sorted_frames:
        return []

    frame_step = _detect_frame_step(sorted_frames)
    total_len = obs_len + pred_len  # 20
    windows = []

    # 滑动窗口遍历每个可能的起始帧
    for start_idx, start_frame in enumerate(sorted_frames):
        # 计算该窗口需要的 20 帧
        window_frames = [
            start_frame + t * frame_step for t in range(total_len)
        ]

        # 检查所有帧是否都存在
        if not all(f in frames for f in window_frames):
            continue

        # 找到在所有 20 帧中都出现的行人
        window_frames_set = set(window_frames)
        common_peds = []
        for ped_id, pf_set in ped_frames.items():
            if window_frames_set.issubset(pf_set):
                common_peds.append(ped_id)

        if len(common_peds) < 1:
            continue

        # 提取轨迹
        obs_traj = np.zeros((len(common_peds), obs_len, 2), dtype=np.float32)
        pred_traj = np.zeros((len(common_peds), pred_len, 2), dtype=np.float32)

        for i, ped_id in enumerate(common_peds):
            for t in range(obs_len):
                obs_traj[i, t] = frames[window_frames[t]][ped_id]
            for t in range(pred_len):
                pred_traj[i, t] = frames[window_frames[obs_len + t]][ped_id]

        windows.append(SceneWindow(
            ped_ids=common_peds,
            obs_traj=obs_traj,
            pred_traj=pred_traj,
            start_frame=start_frame,
        ))

    return windows


def normalize_scene_window(window):
    """
    归一化场景窗口: 每条轨迹减去自身最后一帧观测位置

    原地修改 window.obs_traj 和 window.pred_traj
    """
    origin = window.obs_traj[:, -1:, :]  # (P, 1, 2)
    window.obs_traj = window.obs_traj - origin
    window.pred_traj = window.pred_traj - origin


def build_scene_dataset(data_dir, obs_len=8, pred_len=12, stride=1):
    """
    从数据目录构建场景数据集

    Args:
        data_dir: Social-STGCNN/datasets 目录路径
        obs_len, pred_len: 观测和预测帧数
        stride: 窗口滑动步长 (1 = 每帧都取, 2 = 每隔一帧)
                值越大窗口越少，训练越快但数据越少

    Returns:
        all_windows: list of SceneWindow
    """
    all_windows = []
    txt_files = []

    for root, dirs, files in os.walk(data_dir):
        if "raw" in root or "test-cpu" in root:
            continue
        for f in files:
            if f.endswith('.txt'):
                txt_files.append(os.path.join(root, f))

    for fpath in txt_files:
        frames, ped_frames = load_scene_data(fpath)
        windows = extract_scene_windows(frames, ped_frames, obs_len, pred_len)
        for w in windows:
            normalize_scene_window(w)
        all_windows.extend(windows)

    # 按 stride 采样
    if stride > 1:
        all_windows = all_windows[::stride]

    return all_windows
