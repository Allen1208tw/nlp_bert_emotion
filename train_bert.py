"""
train_bert.py

使用 google-bert/bert-base-chinese 微調「中文句子情緒多元分類」模型。

修改自課堂範例 cases/bert_finetune/train_bert.py（二元分類），
主要差異：

1. NUM_LABELS 由 2 改成 8，id2label / label2id 從 label_map.json 讀取。
2. compute_metrics 改用 accuracy 與 macro-F1（多元分類不能用 average="binary"）。
3. 資料集句子很短（平均約 20 字、最長 92 字），MAX_SEQ_LENGTH 改為 128。
4. 資料量較小（約 3,300 筆 train），learning rate 改為 3e-5，訓練 5 epochs。
5. checkpoints 存在 checkpoints/，最佳模型另存到 output/（不含 checkpoint-*）。

流程：

    data/bert/*.jsonl
        ↓
    Tokenizer + Dynamic Padding
        ↓
    BERT + Sequence Classification Head (8 類)
        ↓
    Cross-Entropy Loss → Backward → Optimizer Step
        ↓
    Validation（以 eval_loss 選最佳 checkpoint，Early Stopping）
        ↓
    output/  ← Best Model
        ↓
    Test set 評估 → output/test_report.txt
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import json
from pathlib import Path

import numpy as np
import torch
import transformers

from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# 2. 模型與路徑設定
# =============================================================================

MODEL_NAME = "google-bert/bert-base-chinese"

DATA_DIR = Path("data/bert")
LABEL_MAP_FILE = DATA_DIR / "label_map.json"

# Trainer 的 checkpoint-* 放這裡（不上傳 GitHub）。
CHECKPOINT_DIR = Path("checkpoints")

# 最佳模型（Best Model）放這裡（上傳 GitHub）。
OUTPUT_DIR = Path("output")

MAX_SEQ_LENGTH = 128


# =============================================================================
# 3. 訓練設定
# =============================================================================

SEED = 42
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 64

# effective batch size = 32 × 1 = 32
#
# train 約 3,327 筆 → 每個 epoch 約 104 個 optimizer steps，
# 5 epochs 約 520 steps。
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 3e-5

# 前 50 個 optimizer steps（約總步數的 10%）做 linear warmup。
#
# 注意：transformers 5.x 已移除 warmup_ratio 參數，只能用 warmup_steps。
WARMUP_STEPS = 50

WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"

# 約每半個 epoch 做一次 validation 與存檔。
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2

EARLY_STOPPING_PATIENCE = 3
LOGGING_STEPS = 10

USE_FP16 = True


# =============================================================================
# 4. 輔助函式
# =============================================================================

def print_gpu_info():
    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"CUDA available：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print("=" * 72)


def load_label_map():
    with open(LABEL_MAP_FILE, encoding="utf-8") as f:
        label_map = json.load(f)

    # JSON 的 key 一定是字串，id2label 要轉回 int。
    id2label = {int(i): label for i, label in label_map["id2label"].items()}
    label2id = {label: int(i) for label, i in label_map["label2id"].items()}
    return id2label, label2id


def compute_metrics(eval_pred):
    """
    多元分類：

        logits shape = [batch_size, 8]
        argmax(axis=-1) → 預測的類別 id

    F1 使用 macro 平均：每個類別各自算 F1 再平均，
    不會因為某些類別筆數多而被主導。
    """

    labels = eval_pred.label_ids
    predictions = eval_pred.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


# =============================================================================
# 5. 主流程
# =============================================================================

def main():

    print_gpu_info()

    for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "label_map.json"):
        if not (DATA_DIR / name).is_file():
            raise FileNotFoundError(
                f"找不到 {DATA_DIR / name}，請先執行 python prepare_data.py"
            )

    id2label, label2id = load_label_map()
    num_labels = len(id2label)
    print(f"類別數：{num_labels} → {list(id2label.values())}")

    # -------------------------------------------------------------------------
    # Step 1：讀取資料
    # -------------------------------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "valid.jsonl"),
            "test": str(DATA_DIR / "test.jsonl"),
        },
    )
    print(dataset)

    # -------------------------------------------------------------------------
    # Step 2：Tokenization（不預先 padding，交給 DataCollator 動態補齊）
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------------------------------------------------------------
    # Step 3：建立 8 類的 BERT sequence classifier
    # -------------------------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # -------------------------------------------------------------------------
    # Step 4：TrainingArguments
    # -------------------------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=USE_FP16 and torch.cuda.is_available(),
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    # -------------------------------------------------------------------------
    # Step 5：Trainer + Early Stopping
    # -------------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    # -------------------------------------------------------------------------
    # Step 6：開始 fine-tuning
    # -------------------------------------------------------------------------

    trainer.train()

    print(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}")
    print(f"最佳 eval_loss：{trainer.state.best_metric}")

    # -------------------------------------------------------------------------
    # Step 7：儲存最佳模型到 output/
    # -------------------------------------------------------------------------
    #
    # load_best_model_at_end=True，此時 trainer.model 已是最佳 checkpoint。

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # trainer_state.json 包含 log_history，plot_bert_metrics.py 會用到。
    trainer.state.save_to_json(str(OUTPUT_DIR / "trainer_state.json"))

    # -------------------------------------------------------------------------
    # Step 8：用 test set 評估最佳模型
    # -------------------------------------------------------------------------

    pred_output = trainer.predict(tokenized_dataset["test"])
    test_preds = np.argmax(pred_output.predictions, axis=-1)
    test_labels = pred_output.label_ids

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(id2label.keys()),
        target_names=list(id2label.values()),
        digits=4,
        zero_division=0,
    )

    print("Test set 評估結果：")
    print(report)

    with open(OUTPUT_DIR / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}\n")
        f.write(f"最佳 eval_loss：{trainer.state.best_metric:.4f}\n\n")
        f.write(report)

    print(f"Final model directory：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
