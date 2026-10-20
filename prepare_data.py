"""
prepare_data.py

讀取 Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset 的 data.csv，
將文字 label 轉成整數 id，並切分成 train / valid / test 三份 JSONL。

輸入：

    data/data.csv           (欄位：text, emotion)

輸出：

    data/bert/train.jsonl
    data/bert/valid.jsonl
    data/bert/test.jsonl
    data/bert/label_map.json   (label ↔ id 對照表)

每一行 JSONL 格式：

    {"text": "誒誒誒！我甄選上了！", "label": 1}
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. 路徑與切分設定
# =============================================================================

RAW_CSV = Path("data/data.csv")
OUTPUT_DIR = Path("data/bert")

SEED = 42

# train : valid : test = 80% : 10% : 10%
VALID_RATIO = 0.1
TEST_RATIO = 0.1


# =============================================================================
# 2. 輔助函式
# =============================================================================

def write_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for text, label in zip(df["text"], df["label"]):
            f.write(
                json.dumps(
                    {"text": text, "label": int(label)},
                    ensure_ascii=False,
                )
                + "\n"
            )


# =============================================================================
# 3. 主流程
# =============================================================================

def main():

    if not RAW_CSV.is_file():
        raise FileNotFoundError(
            f"找不到原始資料：{RAW_CSV.resolve()}\n"
            "請先到 Hugging Face 下載 data.csv 放到 data/ 資料夾。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1：讀取 CSV，清掉空值與前後空白
    # -------------------------------------------------------------------------

    df = pd.read_csv(RAW_CSV, encoding="utf-8")
    df = df.dropna(subset=["text", "emotion"])
    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    print(f"總筆數：{len(df)}")

    # -------------------------------------------------------------------------
    # Step 2：文字 label → 整數 id
    # -------------------------------------------------------------------------
    #
    # 微調時 label 只能是 0, 1, 2, ... 的整數，
    # 所以建立 label2id / id2label 對照表。
    #
    # 依照筆數由多到少排序，讓 id 的順序固定、可重現。
    labels = df["emotion"].value_counts().index.tolist()

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df["label"] = df["emotion"].map(label2id)

    print("Label 對照表：")
    for i, label in id2label.items():
        count = (df["label"] == i).sum()
        print(f"  {i} → {label}（{count} 筆）")

    # -------------------------------------------------------------------------
    # Step 3：分層切分 (stratify)，讓每份資料的類別比例一致
    # -------------------------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=VALID_RATIO + TEST_RATIO,
        stratify=df["label"],
        random_state=SEED,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
        stratify=temp_df["label"],
        random_state=SEED,
    )

    print(
        f"train={len(train_df)}  "
        f"valid={len(valid_df)}  "
        f"test={len(test_df)}"
    )

    # -------------------------------------------------------------------------
    # Step 4：輸出
    # -------------------------------------------------------------------------

    write_jsonl(train_df, OUTPUT_DIR / "train.jsonl")
    write_jsonl(valid_df, OUTPUT_DIR / "valid.jsonl")
    write_jsonl(test_df, OUTPUT_DIR / "test.jsonl")

    with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"輸出完成：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
