"""3클래스 평가: A(기존 2클래스 모델 + 애매하면 중립) vs B(새로 학습한 3클래스 모델).

A: 서비스 중인 INT8 모델(T=2.30 보정)의 긍정 확률이 [lo, hi] 사이면 '중립'.
   lo/hi는 val에서 macro-F1이 최대가 되도록 찾음 (test로 고르면 반칙) → 다시 학습하지 않는 가장 싼 방법의 기준선
B: train_3class.py가 저장한 test 예측(test_preds.tsv)

사용법: python eval_3class.py <data3 폴더> <ko3 모델 폴더>
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, "../backend")
from calibrate import logits_of, probs  # noqa: E402
from sentiment import TEMPERATURE, SentimentModel  # noqa: E402

LABELS = ["부정", "중립", "긍정"]
MIXED = ["배우는 좋은데 스토리가 별로", "스토리는 별로인데 배우는 좋다", "그냥 그랬어요", "보통이에요",
         "나쁘지 않았어요", "좋지도 나쁘지도 않은 영화", "기대보다는 별로였지만 볼만했어요", "한번쯤 볼만한 킬링타임용",
         "최고의 영화, 정말 감동적이에요", "시간이 아까운 최악의 영화"]


def label3(r):
    return 0 if r <= 4 else (1 if r <= 8 else 2)


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows


def report(name, y, pred, ratings):
    print(f"\n### {name}")
    print(f"정확도 {np.mean(pred == y):.2%}")
    print("| 클래스 | 정밀도 | 재현율 | F1 |")
    print("|---|---|---|---|")
    f1s = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum()
        p, r = tp / max((pred == c).sum(), 1), tp / max((y == c).sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        f1s.append(f1)
        print(f"| {LABELS[c]} | {p:.3f} | {r:.3f} | {f1:.3f} |")
    print(f"| **macro 평균** | | | **{np.mean(f1s):.3f}** |")
    print("\n혼동행렬 (행=정답, 열=예측)")
    print("| 정답＼예측 | 부정 | 중립 | 긍정 |")
    print("|---|---|---|---|")
    for c in range(3):
        print(f"| {LABELS[c]} | " + " | ".join(str(((y == c) & (pred == k)).sum()) for k in range(3)) + " |")
    print("\n평점별 예측 분포")
    print("| 평점 | 개수 | 부정 | 중립 | 긍정 |")
    print("|---|---|---|---|---|")
    for rt in range(1, 11):
        s = ratings == rt
        if s.any():
            print(f"| {rt} | {s.sum()} | " + " | ".join(f"{np.mean(pred[s] == k):.0%}" for k in range(3)) + " |")
    return float(np.mean(f1s))


def macro(y, pred):
    out = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum()
        p, r = tp / max((pred == c).sum(), 1), tp / max((y == c).sum(), 1)
        out.append(2 * p * r / max(p + r, 1e-9))
    return np.mean(out)


def band_pred(pos, lo, hi):
    return np.where(pos < lo, 0, np.where(pos > hi, 2, 1))


def main():
    data_dir, ko3_dir = sys.argv[1], sys.argv[2]
    va, te = load(os.path.join(data_dir, "val.tsv")), load(os.path.join(data_dir, "test.tsv"))
    va_y = np.array([label3(int(r["rating"])) for r in va])
    te_y = np.array([label3(int(r["rating"])) for r in te])
    te_rt = np.array([int(r["rating"]) for r in te])

    # ---- A ----
    m2 = SentimentModel(threads=4)
    va_pos = probs(logits_of(m2, [r["text"] for r in va]), TEMPERATURE)[:, 1]
    te_pos = probs(logits_of(m2, [r["text"] for r in te]), TEMPERATURE)[:, 1]
    grid = np.round(np.arange(0.02, 0.99, 0.02), 2)
    best = max(((macro(va_y, band_pred(va_pos, lo, hi)), lo, hi) for lo in grid for hi in grid if hi > lo))
    print(f"A의 중립 구간 (val에서 선택): 긍정 확률 {best[1]:.2f} ~ {best[2]:.2f}  (val macro-F1 {best[0]:.3f})")
    fa = report("A. 기존 2클래스 모델 + 애매하면 중립", te_y, band_pred(te_pos, best[1], best[2]), te_rt)

    # ---- B ----
    preds = load(os.path.join(ko3_dir, "test_preds.tsv"))
    assert [p["text"] for p in preds] == [r["text"] for r in te]
    lg = np.array([[float(p["logit_neg"]), float(p["logit_neu"]), float(p["logit_pos"])] for p in preds])
    fb = report("B. 3클래스로 새로 학습한 모델", te_y, lg.argmax(1), te_rt)

    print(f"\nmacro-F1: A {fa:.3f} → B {fb:.3f}")

    # ---- 애매한 문장 ----
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ko3_dir)
    m3 = AutoModelForSequenceClassification.from_pretrained(ko3_dir).eval()
    with torch.inference_mode():
        p3 = torch.softmax(m3(**tok(MIXED, padding=True, return_tensors="pt")).logits, 1).numpy()
    a_pos = probs(logits_of(m2, MIXED), TEMPERATURE)[:, 1]
    print("\n## 애매한 문장")
    print("| 문장 | A: 긍정 확률 → 판정 | B: 부정 / 중립 / 긍정 → 판정 |")
    print("|---|---|---|")
    for s, a, b in zip(MIXED, a_pos, p3):
        print(f"| {s} | {a:.2f} → {LABELS[band_pred(np.array([a]), best[1], best[2])[0]]} | "
              f"{b[0]:.2f} / {b[1]:.2f} / {b[2]:.2f} → {LABELS[b.argmax()]} |")


if __name__ == "__main__":
    main()
