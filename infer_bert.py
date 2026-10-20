"""
infer_bert.py

載入 train_bert.py 儲存在 output/ 的最佳模型，
對作業指定的句子做情緒分類，輸出格式：

    他們兩個竟然牽手了! => 驚奇語調 (0.9168)

score 是 softmax 後最高類別的機率（confidence），取到小數點後第 4 位。
"""

from pathlib import Path

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


MODEL_DIR = Path("output")

# 作業要求預測的句子。
TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]


def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 train_bert.py。"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # device=0 → 第 0 張 CUDA GPU；device=-1 → CPU
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    results = classifier(TEXTS, truncation=True, max_length=128)

    for text, result in zip(TEXTS, results):
        print(f"{text} => {result['label']} ({result['score']:.4f})")


if __name__ == "__main__":
    main()
