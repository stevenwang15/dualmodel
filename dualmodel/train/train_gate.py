"""
Gate 模型训练脚本。

训练一个 sentence-transformers 编码器 + 小型 MLP 分类头，
用于替代关键词规则路由。

模型结构：
    - 冻结的 sentence-transformers (all-MiniLM-L6-v2) 作为编码器
    - 一个两层的 MLP 分类头

训练流程：
    1. 生成合成训练数据
    2. 划分训练集 / 验证集 (80/20)
    3. 训练 10-20 个 epoch，early stopping
    4. 保存模型到 trained_models/gate_model.pt

使用方式：
    python -m dualmodel.train.train_gate          # 默认训练
    python -m dualmodel.train.train_gate --epochs 20 --num-samples 5000
    python -m dualmodel.train.train_gate --model-source modelscope  # 使用阿里魔搭
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 将项目根目录加入 sys.path，确保能正确导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train.generate_data import generate_gate_data  # noqa: E402

logger = logging.getLogger(__name__)


# ── 编码器加载工具 ──────────────────────────────────────────────

def _load_sentence_encoder(model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
    """从多个源加载 sentence-transformer 编码器。

    加载顺序（从上到下）：
    1. HF 镜像 (hf-mirror.com) — 国内首选，速度快
    2. ModelScope (modelscope.cn) — 国内阿里云源
    3. HuggingFace 直连 — 国外环境
    4. 本地缓存 — 如果之前下过

    Args:
        model_name: sentence-transformers 模型名称。

    Returns:
        加载好的 SentenceTransformer 模型。

    Raises:
        RuntimeError: 所有源都失败时抛出。
    """
    from sentence_transformers import SentenceTransformer

    # ── 方案 A: ModelScope（国内最快，首选） ──
    try:
        from modelscope import snapshot_download
        ms_model_name = f"AI-ModelScope/{model_name}"
        logger.info(f"从 ModelScope 加载: {ms_model_name}")
        model_dir = snapshot_download(ms_model_name)
        model = SentenceTransformer(model_dir)
        logger.info(f"✓ 编码器从 ModelScope 加载")
        return model
    except Exception as e:
        logger.warning(f"ModelScope 加载失败: {e}")

    # ── 方案 B: HF 镜像 ──
    old_hf = os.environ.get("HF_ENDPOINT", "")
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model = SentenceTransformer(model_name)
        logger.info(f"✓ 编码器从 HF 镜像加载: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"HF 镜像加载失败: {e}")
    finally:
        if old_hf:
            os.environ["HF_ENDPOINT"] = old_hf
        else:
            os.environ.pop("HF_ENDPOINT", None)

    # ── 方案 C: HF 直连 ──
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"✓ 编码器从 HuggingFace 直连加载: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"HuggingFace 直连加载失败: {e}")

    # ── 全部失败 ──
    raise RuntimeError(
        f"无法加载编码器模型 {model_name}。\n"
        f"请尝试：\n"
        f"  1. 检查网络连接\n"
        f"  2. 安装 modelscope: pip install modelscope\n"
        f"  3. 手动下载模型到本地缓存"
    )


# ── 模型定义 ─────────────────────────────────────────────────────

class GateModel(nn.Module):
    """Gate 分类模型。

    使用 sentence-transformers 编码文本，然后通过 MLP 分类头
    判断路由到 local (0) 还是 cloud (1)。

    Args:
        embedding_dim: sentence-transformers 输出的 embedding 维度。
        hidden_dim: MLP 隐藏层维度。
        encoder_name: 编码器模型名称。
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 128,
        encoder_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()
        # 加载预训练 sentence-transformer 编码器（多源 fallback）
        self.encoder = _load_sentence_encoder(encoder_name)
        # 冻结 encoder 参数，只训练分类头
        for param in self.encoder.parameters():
            param.requires_grad = False

        # MLP 分类头：embedding → hidden → 2 (local/cloud)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        """前向传播。

        Args:
            texts: 输入文本列表。

        Returns:
            形状为 (batch_size, 2) 的 logits 张量。
        """
        # 编码文本为 embedding（已冻结，不计算梯度）
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        # clone 为普通张量以支持 autograd（encode 返回的可能是 inference-only 张量）
        embeddings = embeddings.clone().to(self.classifier[0].weight.device)

        # 通过分类头
        return self.classifier(embeddings)


# ── 数据集封装 ───────────────────────────────────────────────────

class GateDataset(Dataset):
    """Gate 训练/验证数据集。

    将文本-标签对封装为 PyTorch Dataset。

    Args:
        data: 由 generate_gate_data() 生成的样本列表，
              每项为 {"text": str, "label": int}。
    """

    def __init__(self, data: List[dict]):
        self.texts = [item["text"] for item in data]
        self.labels = [item["label"] for item in data]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def collate_fn(batch: List[Tuple[str, int]]) -> Tuple[List[str], torch.Tensor]:
    """DataLoader 的 collate 函数。

    Args:
        batch: 由 GateDataset.__getitem__ 返回的样本列表。

    Returns:
        (texts, labels) 元组。
    """
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


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
    model: GateModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch。

    Args:
        model: Gate 模型。
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

    for texts, labels in dataloader:
        labels = labels.to(device)

        # 前向传播
        logits = model(texts)
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
    model: GateModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """在验证集上评估模型。

    Args:
        model: Gate 模型。
        dataloader: 验证数据加载器。
        criterion: 损失函数。
        device: 计算设备。

    Returns:
        (平均损失, 准确率) 元组。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for texts, labels in dataloader:
        labels = labels.to(device)
        logits = model(texts)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / num_batches
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main() -> None:
    """主训练入口。

    解析命令行参数，生成数据，训练 Gate 模型，保存权重。
    """
    parser = argparse.ArgumentParser(description="训练 Gate 分类模型")
    parser.add_argument("--epochs", type=int, default=15, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=32, help="batch 大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num-samples", type=int, default=2000, help="训练样本数（不使用 --data-file 时生效）")
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="从 JSON 文件加载训练数据（由 DeepSeek 生成）",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="模型保存目录（默认使用 trained_models/）",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="early stopping 耐心值"
    )
    args = parser.parse_args()

    # ── 设备检测 ──────────────────────────────────────────────────
    device = get_device()
    print(f"{'='*50}")
    print(f"Gate 模型训练")
    print(f"{'='*50}")
    print(f"设备: {device}")

    # ── 数据加载 ──────────────────────────────────────────────────
    if args.data_file:
        import json
        print(f"\n[1/5] 从文件加载训练数据: {args.data_file}")
        with open(args.data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        print(f"  加载 {len(raw_data)} 条数据")
    else:
        print(f"\n[1/5] 生成训练数据 (num_samples={args.num_samples})...")
        raw_data = generate_gate_data(args.num_samples)

    # 划分训练/验证集 (80/20)
    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]

    train_dataset = GateDataset(train_data)
    val_dataset = GateDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(val_dataset)} 条")

    # ── 初始化模型 ────────────────────────────────────────────────
    print(f"\n[2/5] 初始化 Gate 模型 (all-MiniLM-L6-v2)...")
    model = GateModel().to(device)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数量: {sum(p.numel() for p in model.classifier.parameters()):,}")

    # ── 优化器和损失函数 ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # ── 训练循环 ──────────────────────────────────────────────────
    print(f"\n[3/5] 开始训练 (epochs={args.epochs})...")
    best_val_loss = float("inf")
    best_accuracy = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        # 调整学习率
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {lr_now:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping 检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracy = val_accuracy
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

    # ── 最终测试 ──────────────────────────────────────────────────
    print(f"\n[5/5] 最终评估...")
    final_loss, final_acc = evaluate(model, val_loader, criterion, device)
    print(f"  最终验证损失: {final_loss:.4f}")
    print(f"  最终验证准确率: {final_acc:.4f}")

    # 加载最佳模型再次评估
    save_path = _get_save_path(args.save_dir)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        best_loss, best_acc = evaluate(model, val_loader, criterion, device)
        print(f"  最佳模型验证损失: {best_loss:.4f}")
        print(f"  最佳模型验证准确率: {best_acc:.4f}")

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
    return os.path.join(_get_save_dir(custom_dir), "gate_model.pt")


def _save_model(model: GateModel, custom_dir: str | None = None) -> str:
    """保存模型 state_dict。

    只保存 classifier 部分的参数（encoder 是冻结的预训练模型，无需保存）。

    Args:
        model: Gate 模型。
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
