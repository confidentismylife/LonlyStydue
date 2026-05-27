"""
第一步的成果脚本：加载数据 + 可视化 + 最简单的基线预测
纯 CPU 运行，5 分钟出结果

这个脚本的核心目的：让你在 5 分钟内看到一个完整的"数据->预测->评估"流程。
你不需要理解所有细节，先跑通，再做下一步。
"""

import os, sys
# 把项目根目录加入 Python 搜索路径，这样 import src.xxx 才能找到模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== Windows 终端中文乱码的修复 =====
# 默认 Windows 终端用 GBK 编码，Python print 中文会报错/乱码
# 这行把 stdout 编码强行切成 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # 如果 reconfigure 失败（比如被重定向到文件），静默跳过

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Agg 是非交互式后端 — 不弹窗，只把图片存到硬盘
# 你在服务器/核显上跑的时候没有显示器，必须用 Agg
matplotlib.use("Agg")

from src.data_loader import load_eth_ucy_file, extract_trajectories, normalize_trajectories
from src.config import ETH_PATH, UCY_PATH, OBS_LEN, PRED_LEN

# 让 matplotlib 支持中文（否则图上的中文标签会变成方块）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示成方块


def constant_velocity_baseline(obs):
    """
    最简基线模型：匀速直线外推。

    原理：用观测序列最后两帧算出"当前速度向量"，
         然后假设行人按这个速度继续匀速走 12 帧。

    这几乎是你能写出来的最低成本的预测器，
    后续任何深度学习模型都必须比它好，否则就是白训练了。

    参数:
        obs: (N, 8, 2) — N 条轨迹，每条 8 个观测点的 (x, y) 坐标
    返回:
        pred: (N, 12, 2) — N 条预测轨迹，每条 12 个预测点的 (x, y) 坐标
    """
    # 速度向量 = 最后一帧位置 - 倒数第二帧位置，形状 (N, 2)
    vel = obs[:, -1, :] - obs[:, -2, :]
    # 初始化预测数组
    pred = np.zeros((len(obs), PRED_LEN, 2), dtype=np.float32)
    # 逐步外推：第 t 步预测 = 最后一帧 + 速度 * (t+1)
    for t in range(PRED_LEN):
        pred[:, t, :] = obs[:, -1, :] + vel * (t + 1)
    return pred


def visualize_one_sample(obs, pred, gt=None, save_path=None):
    """
    画一条轨迹的"观测 → 预测"对比图。

    蓝线 = 过去 8 帧（模型看到的输入）
    红线 = 未来 12 帧（模型预测的输出）
    绿线 = 真实未来 12 帧（如果给了 gt，用来对比）

    参数:
        obs: (8, 2) 观测坐标
        pred: (12, 2) 预测坐标
        gt: (12, 2) 真实坐标，可选
        save_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 观测轨迹 — 蓝色实线
    ax.plot(obs[:, 0], obs[:, 1], "b-o", linewidth=2, markersize=5, label="观测轨迹 (3.2s)")
    # 起点 — 绿色大圆点
    ax.scatter(obs[0, 0], obs[0, 1], c="green", s=80, zorder=5, label="起点")
    # 当前点（最后一帧观测）— 橙色大圆点
    ax.scatter(obs[-1, 0], obs[-1, 1], c="orange", s=80, zorder=5, label="当前点")

    # 预测轨迹 — 红色虚线
    ax.plot(pred[:, 0], pred[:, 1], "r--o", linewidth=2, markersize=5, label="预测轨迹 (4.8s)")

    # 如果传入了真实轨迹，画出来做参照 — 绿色半透明
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], "g-o", linewidth=2, markersize=3, alpha=0.5, label="真实轨迹")

    ax.set_xlabel("X (米)")
    ax.set_ylabel("Y (米)")
    ax.set_title("行人轨迹预测 Demo\n(匀速基线模型)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")  # X/Y 轴等比例，避免轨迹变形
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")


def main():
    """
    主流程分 4 步：
        [1] 从 Social-STGCNN 目录里找到所有 .txt 数据文件
        [2] 用 data_loader 把每条轨迹切成 (观测8帧, 预测12帧) 的样本
        [3] 归一化后跑匀速基线，算 ADE / FDE
        [4] 随机抽 5 条轨迹画出来
    """
    print("=" * 55)
    print("  火灾疏散项目 - 第一步: 轨迹预测 Demo")
    print("=" * 55)

    # =====================================================================
    # [步骤1] 查找数据
    # 数据在 Social-STGCNN/datasets/ 目录下，按子目录分 train/val/test
    # 这里用 os.walk 递归查找所有 .txt 文件
    # =====================================================================
    print("\n[1/4] 查找数据...")
    # 定位到 Social-STGCNN 的数据目录（和 Huo-zai 平级）
    social_stgcnn_data = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "Social-STGCNN", "datasets"
    )

    # os.walk 递归遍历所有子目录，收集所有 .txt 文件路径
    txt_files = []
    for root, dirs, files in os.walk(social_stgcnn_data):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))

    if not txt_files:
        print("未找到数据文件。请先确保 Social-STGCNN/datasets 中有数据。")
        return

    print(f"找到 {len(txt_files)} 个数据文件")

    # =====================================================================
    # [步骤2] 加载并切分轨迹
    # 每个 .txt 文件格式: frame_id ped_id x y (tab 分隔)
    # load_eth_ucy_file → 按行人 ID 分组
    # extract_trajectories → 滑动窗口切成 (obs=8, pred=12) 的样本
    # =====================================================================
    print("\n[2/4] 加载轨迹数据...")
    all_obs, all_preds = [], []
    for fpath in sorted(txt_files):
        fname = os.path.basename(fpath)
        # 第一层：读取原始文件，按行人 ID 聚合成 {ped_id: [(frame, x, y), ...]}
        raw = load_eth_ucy_file(fpath)
        # 第二层：从每个行人的轨迹中切出 (8帧观测, 12帧预测) 的对
        obs, preds = extract_trajectories(raw)
        if len(obs) > 0:
            all_obs.append(obs)
            all_preds.append(preds)
            print(f"  {fname}: {len(obs)} 条")

    # 把所有文件的数据沿第 0 维拼接 → (总轨迹数, 8, 2) 和 (总轨迹数, 12, 2)
    all_obs = np.concatenate(all_obs)
    all_preds = np.concatenate(all_preds)
    print(f"总计: {len(all_obs)} 条有效轨迹 | obs={all_obs.shape} | pred={all_preds.shape}")

    # =====================================================================
    # [步骤3] 归一化 + 基线预测 + 评估
    # 归一化：每条轨迹的坐标减去自己观测阶段的最后一帧位置
    # 这样做的好处：消除绝对坐标的差异，模型只需要学习"相对位移"
    # =====================================================================
    print("\n[3/4] 运行匀速基线预测...")
    obs_norm, preds_norm = normalize_trajectories(all_obs, all_preds)

    # 固定随机种子，保证每次抽到相同样本（方便复现）
    np.random.seed(42)
    sample_ids = np.random.choice(len(obs_norm), min(5, len(obs_norm)), replace=False)

    # 用匀速外推模型对所有样本做预测
    baseline_preds = constant_velocity_baseline(obs_norm)

    # ---- 两种评估指标 ----
    # ADE (Average Displacement Error): 预测轨迹和真实轨迹之间逐点的平均距离
    # 公式：ADE = mean(||pred_t - gt_t||) 对所有时间步 t 取平均
    ade = np.mean(np.linalg.norm(baseline_preds - preds_norm, axis=2))

    # FDE (Final Displacement Error): 只看最后一帧（最终目的地）的距离
    # 公式：FDE = mean(||pred_last - gt_last||)
    fde = np.mean(np.linalg.norm(baseline_preds[:, -1, :] - preds_norm[:, -1, :], axis=1))

    print(f"\n  [匀速基线模型结果] (全部 {len(obs_norm)} 条轨迹):")
    print(f"     ADE (平均位移误差): {ade:.3f} 米")
    print(f"     FDE (最终位移误差): {fde:.3f} 米")
    print(f"     -> 这个是你的 baseline，Social-LSTM 应该比这个好 30-50%")

    # =====================================================================
    # [步骤4] 可视化 — 随机抽 5 条轨迹画出来
    # 每条图三根线：蓝色观测 / 红色预测 / 绿色真值
    # 图片保存在项目根目录下
    # =====================================================================
    print("\n[4/4] 画图...")
    for i, sid in enumerate(sample_ids):
        obs_i = obs_norm[sid]            # 第 sid 条轨迹的观测
        pred_i = baseline_preds[sid]     # 第 sid 条轨迹的预测
        gt_i = preds_norm[sid]           # 第 sid 条轨迹的真值
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"demo_sample_{i + 1}.png"
        )
        visualize_one_sample(obs_i, pred_i, gt_i, save_path)

    # 收尾输出 — 告诉你刚刚看到了什么
    print("\n" + "=" * 55)
    print("  [完成] 第一步完成！你能看到：")
    print("  1. 蓝色 = 观测到的 8 帧行人轨迹")
    print("  2. 红色 = 匀速模型预测的 12 帧")
    print("  3. 绿色 = 真实轨迹（地面真值）")
    print("")
    print("  [下一步] 用 Social-LSTM 替换匀速模型")
    print("     让 ADE 从 {:.3f} 降到 0.5 以下".format(ade))
    print("=" * 55)


# Python 的约定：当直接运行 python quick_demo.py 时，__name__ 为 "__main__"
# 当被 import 时不会执行 main()，方便被其他脚本复用
if __name__ == "__main__":
    main()
