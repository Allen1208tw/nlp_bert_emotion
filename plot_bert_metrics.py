"""
plot_bert_metrics.py

讀取 train_bert.py 產生的 output/trainer_state.json，輸出：

    images/loss_curve.png       Training / Validation Loss
    images/metrics_curve.png    Validation Accuracy 與 Macro-F1

使用 Matplotlib Agg backend 直接存成 PNG，不需要桌面 GUI。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


STATE_FILE = Path("output/trainer_state.json")
IMAGE_DIR = Path("images")

LOSS_PLOT_FILE = IMAGE_DIR / "loss_curve.png"
METRICS_PLOT_FILE = IMAGE_DIR / "metrics_curve.png"


def collect(logs, key):
    return [
        (item["step"], item[key])
        for item in logs
        if "step" in item and key in item
    ]


def main():

    if not STATE_FILE.is_file():
        raise FileNotFoundError(
            f"找不到：{STATE_FILE.resolve()}\n請先執行 train_bert.py。"
        )

    with STATE_FILE.open("r", encoding="utf-8") as f:
        logs = json.load(f).get("log_history", [])

    if not logs:
        raise RuntimeError("trainer_state.json 中沒有 log_history。")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Training / Validation Loss
    # -------------------------------------------------------------------------

    train_loss = collect(logs, "loss")
    valid_loss = collect(logs, "eval_loss")

    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(*zip(*train_loss), label="Training Loss", linewidth=1.8)
    if valid_loss:
        plt.plot(*zip(*valid_loss), marker="o", label="Validation Loss", linewidth=1.8)
    plt.xlabel("Optimizer Step")
    plt.ylabel("Loss")
    plt.title("BERT Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss 圖表：{LOSS_PLOT_FILE.resolve()}")

    # -------------------------------------------------------------------------
    # Validation Accuracy / Macro-F1
    # -------------------------------------------------------------------------

    eval_acc = collect(logs, "eval_accuracy")
    eval_f1 = collect(logs, "eval_f1_macro")

    plt.figure(figsize=(10, 6))
    if eval_acc:
        plt.plot(*zip(*eval_acc), marker="o", label="Validation Accuracy", linewidth=1.8)
    if eval_f1:
        plt.plot(*zip(*eval_f1), marker="s", label="Validation Macro-F1", linewidth=1.8)
    plt.xlabel("Optimizer Step")
    plt.ylabel("Score")
    plt.title("BERT Validation Accuracy / Macro-F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(METRICS_PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"指標圖表：{METRICS_PLOT_FILE.resolve()}")


if __name__ == "__main__":
    main()
