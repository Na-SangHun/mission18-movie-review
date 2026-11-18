"""3클래스 모델 경량화 + 확률 보정 (lighten.py·calibrate.py의 함수를 재사용).

1) PyTorch FP32 → ONNX FP32 → ONNX INT8(동적 양자화)  : 크기·속도·정확도·McNemar
2) 서버와 같은 전처리 확인: tokenizers(tokenizer.json) ↔ transformers 토큰 일치
3) temperature scaling: val로 T를 맞추고 test로 확인 (ECE, 확신도 구간별 정확도)

조건: 서비스와 같이 리뷰 1개씩(bs=1). 사용법: python lighten_3class.py <data3 폴더> <ko3 모델 폴더>
"""
import os
import statistics
import sys
import time

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from calibrate import fit_T, probs
from eval_3class import LABELS, MIXED, label3, load
from lighten import OUT, export_onnx, mcnemar_p, ort_session


def macro_f1(y, pred):
    f = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum()
        p, r = tp / max((pred == c).sum(), 1), tp / max((y == c).sum(), 1)
        f.append(2 * p * r / max(p + r, 1e-9))
    return float(np.mean(f))


def ece(p, y, bins=10):
    """확신도 구간별 |평균 확신도 - 실제 정확도|의 가중 평균. 3클래스라 확신도는 1/3~1."""
    conf, ok = p.max(1), p.argmax(1) == y
    idx = np.minimum(((conf - 1 / 3) / (2 / 3) * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(conf[idx == b].mean() - ok[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def main():
    data_dir, ko3 = sys.argv[1], sys.argv[2]
    va, te = load(os.path.join(data_dir, "val.tsv")), load(os.path.join(data_dir, "test.tsv"))
    va_x, te_x = [r["text"] for r in va], [r["text"] for r in te]
    va_y = np.array([label3(int(r["rating"])) for r in va])
    te_y = np.array([label3(int(r["rating"])) for r in te])

    hf_tok = AutoTokenizer.from_pretrained(ko3)
    tok = Tokenizer.from_file(os.path.join(ko3, "tokenizer.json"))  # 서버가 쓸 가벼운 토크나이저
    tok.enable_truncation(128)
    bad = sum(hf_tok(t, truncation=True, max_length=128)["input_ids"] != tok.encode(t).ids for t in te_x)
    print(f"토크나이저 불일치: {bad} / {len(te_x)}")

    def feed(t):
        e = tok.encode(t)
        return {"input_ids": np.array([e.ids], np.int64), "attention_mask": np.array([e.attention_mask], np.int64),
                "token_type_ids": np.array([e.type_ids], np.int64)}

    fp32 = AutoModelForSequenceClassification.from_pretrained(ko3).eval()
    p_fp32, p_int8 = os.path.join(OUT, "ko3_fp32.onnx"), os.path.join(OUT, "ko3_int8.onnx")
    export_onnx(fp32, hf_tok, p_fp32)
    quantize_dynamic(onnx.shape_inference.infer_shapes(onnx.load(p_fp32)), p_int8, weight_type=QuantType.QInt8)

    @torch.inference_mode()
    def run_torch(t):
        return fp32(**{k: torch.from_numpy(v) for k, v in feed(t).items()}).logits.numpy()[0]

    s32, s8 = ort_session(p_fp32), ort_session(p_int8)
    runners = {"PyTorch FP32": run_torch,
               "ONNX FP32": lambda t: s32.run(["logits"], feed(t))[0][0],
               "ONNX INT8": lambda t: s8.run(["logits"], feed(t))[0][0]}
    sizes = {"PyTorch FP32": sum(p.numel() * 4 for p in fp32.parameters()) / 2**20,
             "ONNX FP32": os.path.getsize(p_fp32) / 2**20, "ONNX INT8": os.path.getsize(p_int8) / 2**20}

    logits = {k: np.array([run(t) for t in te_x]) for k, run in runners.items()}
    print(f"PyTorch↔ONNX FP32 logit 최대 차이: {np.abs(logits['PyTorch FP32'] - logits['ONNX FP32']).max():.2e}")

    lat = {k: [] for k in runners}
    for run in runners.values():
        for t in te_x[:20]:
            run(t)  # warm-up
    for _ in range(3):  # 번갈아 3라운드 × 200문장, 중앙값
        for k, run in runners.items():
            for t in te_x[:200]:
                s = time.perf_counter()
                run(t)
                lat[k].append((time.perf_counter() - s) * 1000)

    ref_ok = logits["PyTorch FP32"].argmax(1) == te_y
    print("\n| 변형 | 크기(MB) | 정확도 | macro-F1 | McNemar vs FP32 (b/c, p) | 예측 다름 | 지연 중앙값(ms, bs=1) |")
    print("|---|---|---|---|---|---|---|")
    for k, lg in logits.items():
        pred = lg.argmax(1)
        b, c, p = mcnemar_p(ref_ok, pred == te_y)
        print(f"| {k} | {sizes[k]:.1f} | {np.mean(pred == te_y):.2%} | {macro_f1(te_y, pred):.4f} | {b}/{c}, p={p:.2f} "
              f"| {(pred != logits['PyTorch FP32'].argmax(1)).sum()} | {statistics.median(lat[k]):.2f} |")

    # 보정: val로 T 맞추기 (서비스용 INT8 logits). test는 확인용으로만 사용
    va_lg = np.array([runners["ONNX INT8"](t) for t in va_x])
    T = fit_T(va_lg, va_y)
    lg8 = logits["ONNX INT8"]
    print(f"\nT (val 9,000개로 맞춤) = {T:.3f}")
    print("| | test NLL | test ECE |")
    print("|---|---|---|")
    for name, t_ in (("보정 전 T=1", 1.0), (f"보정 후 T={T:.2f}", T)):
        p = probs(lg8, t_)
        print(f"| {name} | {-np.log(p[np.arange(len(te_y)), te_y] + 1e-12).mean():.4f} | {ece(p, te_y):.4f} |")
    for name, t_ in (("보정 전", 1.0), ("보정 후", T)):
        p = probs(lg8, t_)
        conf, ok = p.max(1), p.argmax(1) == te_y
        print(f"\n확신도 구간별 실제 정확도 ({name})\n| 구간 | 문장 수 | 평균 확신도 | 실제 정확도 |\n|---|---|---|---|")
        for lo, hi in ((1 / 3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
            s = (conf >= lo) & (conf < hi)
            if s.any():
                print(f"| {lo:.2f}~{min(hi, 1):.2f} | {s.sum()} | {conf[s].mean():.1%} | {ok[s].mean():.1%} |")

    print(f"\n애매한 문장 (ONNX INT8, T={T:.2f}): 부정 / 중립 / 긍정")
    for s in MIXED:
        p = probs(runners["ONNX INT8"](s)[None], T)[0]
        print(f"- {s}: {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f} → {LABELS[p.argmax()]}")


if __name__ == "__main__":
    main()
