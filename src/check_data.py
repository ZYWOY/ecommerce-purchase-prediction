from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target_path = project_root / "data" / "processed" / "y_purchase.npy"

    if not target_path.exists():
        raise FileNotFoundError(f"未找到目标文件：{target_path}")

    y = np.load(target_path)

    print(f"总样本数：{len(y)}")
    print("标签统计：")
    print(f"  未购买样本：{int((y == 0).sum())}")
    print(f"  购买样本：{int((y == 1).sum())}")
    print(f"  正样本占比：{float(y.mean()):.4f}")


if __name__ == "__main__":
    main()
