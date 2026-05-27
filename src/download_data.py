"""下载 ETH/UCY 数据集 — 约 500MB，纯 CPU"""

import os
import urllib.request
import zipfile
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_RAW, ETH_PATH, UCY_PATH


def download_file(url, dest):
    """带进度条的下载"""
    print(f"下载: {url}")
    print(f"保存到: {dest}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        bar = "=" * (int(percent) // 2)
        print(f"\r[{bar:<50}] {percent:.1f}%", end="")

    urllib.request.urlretrieve(url, dest, progress)
    print()


def main():
    # 检查数据是否已存在
    eth_exists = os.path.exists(ETH_PATH) and len(os.listdir(ETH_PATH)) > 0
    ucy_exists = os.path.exists(UCY_PATH) and len(os.listdir(UCY_PATH)) > 0

    if eth_exists and ucy_exists:
        print("数据集已存在，跳过下载。")
        return

    # 下载
    zip_path = os.path.join(DATA_RAW, "ethucy.zip")
    download_file("https://github.com/crowdbotp/OpenTraj/raw/master/datasets/ETH_UCY/ethucy.zip", zip_path)

    # 解压
    print("解压中...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW)
    print("解压完成。")

    # 清理 zip
    os.remove(zip_path)
    print(f"ETH 数据路径: {ETH_PATH}")
    print(f"UCY 数据路径: {UCY_PATH}")


if __name__ == "__main__":
    main()
