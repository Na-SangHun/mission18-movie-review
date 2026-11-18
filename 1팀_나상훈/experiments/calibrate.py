"""확률 보정 (temperature scaling): p = softmax(logits / T)

T로 나누기만 하므로 argmax(예측)·정확도는 그대로이고, 확률의 '과신'만 줄인다.
T는 NLL(정답 확률의 -log 평균)이 최소가 되도록 1차원 탐색으로 찾는다.

데이터 (서로 겹치지 않음)
  - 평가: NSMC test 시드42 셔플의 앞 2,000개 (다른 실험과 같은 샘플)
  - 보정(권장): 같은 셔플의 2,000~5,000번째 3,000개 → 모델 학습에 쓰이지 않은 데이터
  - 보정(비교용): NSMC train 3,000개 → 모델이 학습한 데이터라 T가 작게(과신 그대로) 나올 것이라는 가설 확인

사용법: python calibrate.py <ratings_test.txt> <ratings_train.txt>
"""
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, "../backend")
from select_model import load_nsmc  # noqa: E402
from sentiment import SentimentModel  # noqa: E402

MIXED = ["배우는 좋은데 스토리가 별로", "스토리는 별로인데 배우는 좋다", "그냥 그랬어요", "보통이에요",
         "나쁘지 않았어요", "좋지도 나쁘지도 않은 영화", "최고의 영화, 정말 감동적이에요", "시간이 아까운 최악의 영화"]


def logits_of(m, texts):
    """서비스 조건(INT8, 1개씩)의 logits. T 보정 전 원래 값."""
    out = []
    for t in texts:
        enc = m.tok.encode(t)
        feed = {k: np.array([v], dtype=np.int64) for k, v in
                (("input_ids", enc.ids), ("attention_mask", enc.attention_mask), ("token_type_ids", enc.type_ids))}
        out.append(m.sess.run(["logits"], feed)[0][0])
    return np.array(out)


def probs(logits, T=1.0):
    z = logits / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def nll(logits, y, T):
    return -np.log(probs(logits, T)[np.arange(len(y)), y] + 1e-12).mean()


def fit_T(logits, y):
    return minimize_scalar(lambda T: nll(logits, y, T), bounds=(0.05, 20), method="bounded").x


def ece(p, y, bins=10):
    """Expected Calibration Error: 확신도 구간별 |확신도 평균 - 실제 정확도|의 가중 평균. 0이면 완벽."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        if s.any():
            total += s.mean() * abs(conf[s].mean() - (pred[s] == y[s]).mean())
    return total


def table(p, y):
    conf, ok = p.max(1), p.argmax(1) == y
    rows = []
    for lo, hi in ((.5, .7), (.7, .9), (.9, .99), (.99, 1.01)):
        s = (conf >= lo) & (conf < hi)
        rows.append(f"| {lo:.2f}~{min(hi, 1):.2f} | {s.sum()} | {conf[s].mean() if s.any() else 0:.1%} "
                    f"| {ok[s].mean() if s.any() else 0:.1%} |")
    return "\n".join(rows)


def main():
    test_path, train_path = sys.argv[1], sys.argv[2]
    texts, labels = load_nsmc(test_path, 5000)
    ev_x, ev_y = texts[:2000], np.array(labels[:2000])
    cal_x, cal_y = texts[2000:5000], np.array(labels[2000:5000])
    tr_x, tr_y = load_nsmc(train_path, 3000)
    tr_y = np.array(tr_y)

    m = SentimentModel(threads=4)
    L_ev, L_cal, L_tr = logits_of(m, ev_x), logits_of(m, cal_x), logits_of(m, tr_x)

    T_cal, T_tr = fit_T(L_cal, cal_y), fit_T(L_tr, tr_y)
    acc_tr = (L_tr.argmax(1) == tr_y).mean()
    print(f"T (test 보정용 3,000개, 학습에 안 쓰인 데이터) = {T_cal:.3f}")
    print(f"T (train 3,000개, 학습에 쓰인 데이터)          = {T_tr:.3f}  (train 정확도 {acc_tr:.2%})\n")

    print("## 평가셋 2,000개: 보정 전/후")
    print("| T | 정확도 | NLL | ECE |")
    print("|---|---|---|---|")
    for name, T in (("1 (보정 전)", 1.0), (f"{T_tr:.2f} (train으로 맞춤)", T_tr), (f"{T_cal:.2f} (test 보정용으로 맞춤)", T_cal)):
        p = probs(L_ev, T)
        print(f"| {name} | {(p.argmax(1) == ev_y).mean():.2%} | {nll(L_ev, ev_y, T):.4f} | {ece(p, ev_y):.4f} |")

    for name, T in (("보정 전 (T=1)", 1.0), (f"보정 후 (T={T_cal:.2f})", T_cal)):
        print(f"\n### 확신도 구간별 실제 정확도, {name}")
        print("| 확신도 구간 | 문장 수 | 평균 확신도 | 실제 정확도 |")
        print("|---|---|---|---|")
        print(table(probs(L_ev, T), ev_y))

    print("\n## 애매한 문장의 긍정 확률: 보정 전 → 후")
    print("| 문장 | T=1 | 보정 후 |")
    print("|---|---|---|")
    L_mx = logits_of(m, MIXED)
    for s, a, b in zip(MIXED, probs(L_mx)[:, 1], probs(L_mx, T_cal)[:, 1]):
        print(f"| {s} | {a:.3f} | {b:.3f} |")


if __name__ == "__main__":
    main()
