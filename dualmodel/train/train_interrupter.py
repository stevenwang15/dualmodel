"""
Interrupter 模型训练脚本。

训练一个小型 MLP 序列分类器，输入 token 特征窗口，
输出是否应该触发模型切换。

输入特征：
    每个样本是一个长度为 window_size * 2 的向量，
    包含连续的 (confidence, entropy) 对。

输出：
    logits: [continue_prob, switch_prob]

训练流程：
    1. 生成合成序列数据
    2. 划分训练集 / 验证集 (80/20)
    3. 训练 20-30 个 epoch，early stopping
    4. 评估准确率、精确率、召回率
    5. 保存模型到 trained_models/interrupter_model.pt

使用方式：
    python -m dualmodel.train.train_interrupter           # 默认训练
    python -m dualmodel.train.train_interrupter --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 将项目根目录加入 sys.path，确保能正确导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train.generate_data import generate_interrupter_data  # noqa: E402


# ── 模型定义 ─────────────────────────────────────────────────────

class InterrupterModel(nn.Module):
    """Interrupter 序列分类模型。

    输入 token 级特征窗口（confidence, entropy 对），
    输出是否应触发模型切换的决策。

    Args:
        window_size: 滑动窗口大小。
        hidden_dim: MLP 隐藏层维度。
    """

    def __init__(self, window_size: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.window_size = window_size
        input_dim = window_size * 2  # confidence + entropy for each token

        # 小型 MLP 分类器
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # continue (0), switch (1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状为 (batch_size, window_size * 2) 的输入张量。

        Returns:
            形状为 (batch_size, 2) 的 logits 张量。
        """
        return self.net(x)


# ── 数据集封装 ───────────────────────────────────────────────────

class InterrupterDataset(Dataset):
    """Interrupter 训练/验证数据集。

    Args:
        X: 形状为 (num_samples, window_size * 2) 的特征数组。
        y: 形状为 (num_samples,) 的标签数组（0=继续, 1=切换）。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── 训练流程 ─────────────────────────────────────────────────────

def get_device() -> torch.device:
    """自动检测可用设备。

    优先级: mps > cuda > cpu。

    Returns:
        torch.device 对象。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: InterrupterModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch。

    Args:
        model: Interrupter 模型。
        dataloader: 训练数据加载器。
        optimizer: 优化器。
        criterion: 损失函数。
        device: 计算设备。

    Returns:
        本 epoch 的平均损失。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        # 前向传播
        logits = model(features)
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: InterrupterModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """在验证集上评估模型。

    计算损失、准确率、精确率、召回率。

    Args:
        model: Interrupter 模型。
        dataloader: 验证数据加载器。
        criterion: 损失函数。
        device: 计算设备。

    Returns:
        (平均损失, 准确率, 精确率, 召回率) 元组。
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds: list[int] = []
    all_labels: list[int] = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / num_batches

    # 计算指标
    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)

    accuracy = (all_preds_arr == all_labels_arr).mean()

    # 精确率: TP / (TP + FP)
    tp = ((all_preds_arr == 1) & (all_labels_arr == 1)).sum()
    fp = ((all_preds_arr == 1) & (all_labels_arr == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 召回率: TP / (TP + FN)
    fn = ((all_preds_arr == 0) & (all_labels_arr == 1)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return avg_loss, float(accuracy), float(precision), float(recall)


def main() -> None:
    """主训练入口。

    解析命令行参数，生成数据，训练 Interrupter 模型，保存权重。
    """
    parser = argparse.ArgumentParser(description="训练 Interrupter 序列分类模型")
    parser.add_argument("--epochs", type=int, default=25, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=64, help="batch 大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num-samples", type=int, default=5000, help="训练样本数")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="模型保存目录（默认使用 trained_models/）",
    )
    parser.add_argument(
        "--patience", type=int, default=7, help="early stopping 耐心值"
    )
    parser.add_argument(
        "--window-size", type=int, default=5, help="滑动窗口大小"
    )
    args = parser.parse_args()

    # ── 设备检测 ──────────────────────────────────────────────────
    device = get_device()
    print(f"{'='*50}")
    print(f"Interrupter 模型训练")
    print(f"{'='*50}")
    print(f"设备: {device}")

    # ── 生成数据 ──────────────────────────────────────────────────
    print(f"\n[1/5] 生成训练数据 (num_samples={args.num_samples})...")
    X_raw, y_raw = generate_interrupter_data(args.num_samples)

    # 划分训练/验证集 (80/20)
    split_idx = int(len(X_raw) * 0.8)
    X_train, X_val = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_val = y_raw[:split_idx], y_raw[split_idx:]

    train_dataset = InterrupterDataset(X_train, y_train)
    val_dataset = InterrupterDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
    )

    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(val_dataset)} 条")
    print(f"  特征维度: {X_raw.shape[1]}")

    # ── 初始化模型 ────────────────────────────────────────────────
    print(f"\n[2/5] 初始化 Interrupter 模型...")
    model = InterrupterModel(
        window_size=args.window_size,
        hidden_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # ── 优化器和损失函数 ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # ── 训练循环 ──────────────────────────────────────────────────
    print(f"\n[3/5] 开始训练 (epochs={args.epochs})...")
    best_val_loss = float("inf")
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss, val_acc, val_prec, val_rec = evaluate(
            model, val_loader, criterion, device,
        )

        # 调整学习率
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Prec: {val_prec:.4f} | "
            f"Rec: {val_rec:.4f} | "
            f"LR: {lr_now:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping 检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracy = val_acc
            best_precision = val_prec
            best_recall = val_rec
            patience_counter = 0
            # 保存最佳模型
            _save_model(model, args.save_dir)
            print(f"    → 验证损失下降，保存模型")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    → Early stopping (连续 {args.patience} 个 epoch 未改善)")
                break

    total_time = time.time() - start_time
    print(f"\n[4/5] 训练完成！总耗时: {total_time:.1f}s")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  最佳验证准确率: {best_accuracy:.4f}")
    print(f"  最佳验证精确率: {best_precision:.4f}")
    print(f"  最佳验证召回率: {best_recall:.4f}")

    # ── 最终测试 ──────────────────────────────────────────────────
    print(f"\n[5/5] 最终评估...")
    # 加载最佳模型
    save_path = _get_save_path(args.save_dir)
    if os.path.exists(save_path):
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )
        final_loss, final_acc, final_prec, final_rec = evaluate(
            model, val_loader, criterion, device,
        )
        print(f"  最佳模型验证损失: {final_loss:.4f}")
        print(f"  最佳模型验证准确率: {final_acc:.4f}")
        print(f"  最佳模型验证精确率: {final_prec:.4f}")
        print(f"  最佳模型验证召回率: {final_rec:.4f}")

    print(f"\n{'='*50}")
    print(f"训练完成！模型已保存到: {save_path}")
    print(f"{'='*50}")


def _get_save_dir(custom_dir: str | None = None) -> str:
    """获取模型保存目录的绝对路径。

    Args:
        custom_dir: 用户自定义的保存目录，为 None 时使用默认路径。

    Returns:
        保存目录的绝对路径。
    """
    if custom_dir:
        save_dir = os.path.abspath(custom_dir)
    else:
        # 默认：train/ 的父目录下的 trained_models/
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "trained_models",
        )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _get_save_path(custom_dir: str | None = None) -> str:
    """获取模型文件的完整保存路径。

    Args:
        custom_dir: 自定义保存目录。

    Returns:
        模型文件的完整路径。
    """
    return os.path.join(_get_save_dir(custom_dir), "interrupter_model.pt")


def _save_model(model: InterrupterModel, custom_dir: str | None = None) -> str:
    """保存模型 state_dict。

    Args:
        model: Interrupter 模型。
        custom_dir: 自定义保存目录。

    Returns:
        保存的文件路径。
    """
    save_path = _get_save_path(custom_dir)
    torch.save(model.state_dict(), save_path)
    print(f"  模型已保存到: {save_path}")
    return save_path


if __name__ == "__main__":
    main()
