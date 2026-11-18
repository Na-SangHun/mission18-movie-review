"""FP32가 확신(≥0.9)했는데 INT8(E)에서 뒤집힌 문장의 원인 조사.

가설
  H1. 묶음(batch) 효과: DynamicQuantizeLinear는 텐서 전체(묶음 안의 모든 문장·토큰)에서
      scale을 하나 계산함 → 같은 묶음의 다른 문장이 이 문장의 양자화 정밀도를 바꿀 수 있음
  H2. 문장 길이/섞인 감정: 길고 칭찬·비판이 섞인 문장 자체가 양자화에 약함
  H3. 특정 층: 일부 층의 양자화 오차가 결과를 뒤집음

사용법: python investigate_flips.py <ratings_test.txt 경로>
"""
import os
import sys

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from lighten import MODEL_ID, OUT, encode, ort_session, softmax
from select_model import load_nsmc

BS = 64  # lighten.py 정확도 측정 때의 묶음 크기


def probs_batched(sess, tok, texts, bs):
    run = lambda t: softmax(sess.run(["logits"], encode(tok, t))[0])[:, 1]
    return np.concatenate([run(texts[i:i + bs]) for i in range(0, len(texts), bs)])


def main():
    texts, labels = load_nsmc(sys.argv[1], 2000)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    fp32 = ort_session(os.path.join(OUT, "model_fp32.onnx"))
    int8 = ort_session(os.path.join(OUT, "model_int8.onnx"))

    p32 = probs_batched(fp32, tok, texts, BS)        # FP32는 묶음과 무관 (확인용으로 bs=1과 비교)
    p32_1 = probs_batched(fp32, tok, texts, 1)
    p8_b = probs_batched(int8, tok, texts, BS)       # lighten.py와 같은 조건
    p8_1 = probs_batched(int8, tok, texts, 1)        # 서비스 조건: 리뷰 1개씩
    print(f"FP32 bs=64 vs bs=1 최대 차이: {np.abs(p32 - p32_1).max():.2e} (FP32는 묶음 영향 없음 확인)")

    sure = np.maximum(p32, 1 - p32) >= 0.9
    flip_b = (p32 > .5) != (p8_b > .5)
    flip_1 = (p32 > .5) != (p8_1 > .5)
    acc = lambda p: np.mean((p > .5) == np.array(labels, bool))

    print("\n## H1. 묶음 효과: 같은 INT8 모델을 묶음 64 vs 1개씩 실행")
    print("| 조건 | 정확도 | FP32와 예측 다름 | 그중 FP32 확신≥0.9 | 긍정확률 차이 평균/최대 |")
    print("|---|---|---|---|---|")
    for name, p, f in (("INT8 bs=64", p8_b, flip_b), ("INT8 bs=1", p8_1, flip_1)):
        g = np.abs(p - p32)
        print(f"| {name} | {acc(p):.2%} | {f.sum()} | {(f & sure).sum()} | {g.mean():.4f}/{g.max():.4f} |")

    print("\n확신 문장이 뒤집힌 경우 (bs=64): 문장 / 토큰 수 / FP32 / INT8 bs=64 / INT8 bs=1 / 묶음 최대 토큰 수")
    for i in np.where(flip_b & sure)[0]:
        start = i // BS * BS
        max_len = max(len(tok(t, truncation=True, max_length=128)["input_ids"]) for t in texts[start:start + BS])
        n_tok = len(tok(texts[i], truncation=True, max_length=128)["input_ids"])
        print(f"- {texts[i]!r} / {n_tok} / {p32[i]:.2f} / {p8_b[i]:.2f} / {p8_1[i]:.2f} / {max_len}")

    # H1 추가 확인: 묶음 효과가 '패딩 길이'와 '다른 문장의 값' 중 어느 쪽인지
    #  - 같은 문장 1개를 패딩만 붙여 128 토큰으로 늘려 실행 (다른 문장 없음)
    print("\n패딩만 늘린 경우 (다른 문장 없이 max_length=128 패딩)")
    for i in np.where(flip_b & sure)[0]:
        enc = tok([texts[i]], padding="max_length", truncation=True, max_length=128, return_tensors="np")
        enc = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")}
        print(f"- {texts[i][:25]!r}: INT8 pad128 = {softmax(int8.run(['logits'], enc)[0])[0, 1]:.2f}")

    print("\n## H2. 길이 효과 (bs=1, 서비스 조건): 토큰 길이 구간별 뒤집힘")
    lens = np.array([len(tok(t, truncation=True, max_length=128)["input_ids"]) for t in texts])
    print("| 토큰 길이 | 문장 수 | 뒤집힘(bs=1) | 뒤집힘(bs=64) |")
    print("|---|---|---|---|")
    for lo, hi in ((0, 16), (16, 32), (32, 64), (64, 129)):
        m = (lens >= lo) & (lens < hi)
        print(f"| {lo}~{hi - 1} | {m.sum()} | {flip_1[m].sum()} | {flip_b[m].sum()} |")

    # H3. 층별 민감도: 한 층만 FP32로 남기고(나머지는 INT8) bs=1 오차가 얼마나 줄어드는지
    print("\n## H3. 층별 민감도 (해당 층만 양자화 제외, bs=1, 앞 500문장)")
    model = onnx.shape_inference.infer_shapes(onnx.load(os.path.join(OUT, "model_fp32.onnx")))
    matmuls = [n.name for n in model.graph.node if n.op_type == "MatMul"]
    groups = {"embeddings_project": [m for m in matmuls if "embeddings_project" in m]}
    for L in range(12):
        groups[f"layer.{L}"] = [m for m in matmuls if f"layer.{L}/" in m]
    groups["classifier"] = [m for m in matmuls if "classifier" in m]
    sub = list(range(500))
    base_gap = np.abs(p8_1[sub] - p32[sub]).mean()
    print(f"전체 INT8 기준 오차 평균: {base_gap:.4f}")
    print("| FP32로 남긴 층 | 긍정확률 오차 평균 | 기준 대비 감소율 |")
    print("|---|---|---|")
    tmp = os.path.join(OUT, "tmp_excl.onnx")
    for g, nodes in groups.items():
        quantize_dynamic(onnx.shape_inference.infer_shapes(onnx.load(os.path.join(OUT, "model_fp32.onnx"))),
                         tmp, weight_type=QuantType.QInt8, nodes_to_exclude=nodes)
        p = probs_batched(ort_session(tmp), tok, [texts[i] for i in sub], 1)
        gap = np.abs(p - p32[sub]).mean()
        print(f"| {g} | {gap:.4f} | {1 - gap / base_gap:.0%} |")

    # H3 후속: 서비스 조건(bs=1)에서 확신 문장이 뒤집힌 목록 + 민감했던 layer.0·1을 FP32로 남긴 혼합 정밀도 모델
    print("\n서비스 조건(bs=1)에서 확신 문장이 뒤집힌 경우: 문장 / FP32 / INT8")
    for i in np.where(flip_1 & sure)[0]:
        print(f"- {texts[i]!r} / {p32[i]:.2f} / {p8_1[i]:.2f}")
    mixed = os.path.join(OUT, "model_int8_mixed.onnx")
    quantize_dynamic(onnx.shape_inference.infer_shapes(onnx.load(os.path.join(OUT, "model_fp32.onnx"))),
                     mixed, weight_type=QuantType.QInt8, nodes_to_exclude=groups["layer.0"] + groups["layer.1"])
    pm = probs_batched(ort_session(mixed), tok, texts, 1)
    fm = (p32 > .5) != (pm > .5)
    print("\n| 모델 (bs=1, 2000문장) | 크기(MB) | 정확도 | FP32와 예측 다름 | 그중 확신≥0.9 | 오차 평균 |")
    print("|---|---|---|---|---|---|")
    for name, path, p, f in (("INT8 전체", os.path.join(OUT, "model_int8.onnx"), p8_1, flip_1),
                             ("INT8 + layer.0·1 FP32", mixed, pm, fm)):
        print(f"| {name} | {os.path.getsize(path) / 2**20:.1f} | {acc(p):.2%} | {f.sum()} | {(f & sure).sum()} "
              f"| {np.abs(p - p32).mean():.4f} |")
    os.remove(tmp)


if __name__ == "__main__":
    main()
