"""경량화 실험: PyTorch → ONNX → INT8 동적 양자화.

원인을 나눠서 측정하기 위해(미션 16 피드백) 한 번에 한 가지만 바꾼 변형을 만든다.
  A. PyTorch FP32                     ← 기준
  B. PyTorch INT8 (동적 양자화)       ← A와 비교: PyTorch 안에서 양자화만의 효과
  C. ORT FP32, 그래프 최적화 끔       ← A와 비교: 실행 엔진(PyTorch→ORT)만의 효과
  D. ORT FP32, 그래프 최적화 켬       ← C와 비교: 그래프 최적화(연산 합치기)만의 효과
  E. ORT INT8 동적 양자화             ← D와 비교: ORT에서 양자화만의 효과
  F. ORT INT8 동적 양자화 reduce_range ← E와 비교: VNNI 없는 CPU 포화 대책 효과
  G. F + per_channel                  ← F와 비교: 채널별 scale 효과

사용법: python lighten.py <ratings_test.txt 경로> [N]
"""
import math
import os
import random
import statistics
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from select_model import load_nsmc  # 모델 선정 때와 같은 샘플(시드 42)로 평가

MODEL_ID = "daekeun-ml/koelectra-small-v3-nsmc"
OUT = os.path.join(os.path.dirname(__file__), "models")
THREADS = 4
torch.set_num_threads(THREADS)
INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


def export_onnx(model, tok, path):
    """PyTorch 모델을 ONNX 그래프로 저장한다. batch·문장 길이는 바뀔 수 있게 dynamic으로 둔다."""
    enc = tok("샘플 문장", return_tensors="pt")
    axes = {k: {0: "batch", 1: "seq"} for k in INPUTS} | {"logits": {0: "batch"}}
    torch.onnx.export(model, tuple(enc[k] for k in INPUTS), path, dynamo=False,
                      input_names=INPUTS, output_names=["logits"],
                      dynamic_axes=axes, opset_version=17)


def ort_session(path, optimize=True):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize
                                   else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def make_runners(tok):
    """변형마다 '토큰화된 입력(dict of np.int64) → logits(np)' 함수를 만든다."""
    os.makedirs(OUT, exist_ok=True)
    fp32 = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    # 동적 양자화: Linear 층 가중치만 INT8로 저장, 활성값은 실행 중에 범위를 재서 양자화
    torch_int8 = torch.ao.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)

    p_fp32 = os.path.join(OUT, "model_fp32.onnx")
    p_int8 = os.path.join(OUT, "model_int8.onnx")
    p_int8_rr = os.path.join(OUT, "model_int8_rr.onnx")
    export_onnx(fp32, tok, p_fp32)
    # 경로 대신 모델 객체를 넘김: 경로를 넘기면 내부 C++ shape inference가 한글 경로를 못 읽어
    # '-inferred.onnx' 파일이 안 생기고 FileNotFoundError가 남 (트러블슈팅 참고)
    proto = lambda: onnx.shape_inference.infer_shapes(onnx.load(p_fp32))
    p_int8_pc = os.path.join(OUT, "model_int8_pc.onnx")
    quantize_dynamic(proto(), p_int8, weight_type=QuantType.QInt8)
    # reduce_range: 가중치를 7bit 범위로 줄여 AVX2(VNNI 없음) 커널의 16bit 중간합 포화를 피함
    # (주의: extra_options가 아니라 함수 인자로 넘겨야 적용됨 — 처음엔 잘못 넣어 E와 결과가 같았음)
    quantize_dynamic(proto(), p_int8_rr, weight_type=QuantType.QInt8, reduce_range=True)
    # per_channel: 출력 채널마다 scale을 따로 둬서, 값 범위가 큰 채널 때문에 나머지가 뭉개지는 것을 막음
    quantize_dynamic(proto(), p_int8_pc, weight_type=QuantType.QInt8, reduce_range=True, per_channel=True)

    def torch_runner(m):
        @torch.inference_mode()
        def run(enc):
            return m(**{k: torch.from_numpy(v) for k, v in enc.items()}).logits.numpy()
        return run

    def ort_runner(sess):
        return lambda enc: sess.run(["logits"], enc)[0]

    runners = {
        "A. PyTorch FP32": torch_runner(fp32),
        "B. PyTorch INT8": torch_runner(torch_int8),
        "C. ORT FP32 (최적화 끔)": ort_runner(ort_session(p_fp32, optimize=False)),
        "D. ORT FP32 (최적화 켬)": ort_runner(ort_session(p_fp32)),
        "E. ORT INT8": ort_runner(ort_session(p_int8)),
        "F. ORT INT8 reduce_range": ort_runner(ort_session(p_int8_rr)),
        "G. ORT INT8 reduce_range+per_channel": ort_runner(ort_session(p_int8_pc)),
    }
    torch_mb = lambda m: sum(t.numel() * t.element_size() for t in m.state_dict().values()
                             if isinstance(t, torch.Tensor)) / 2**20
    sizes = {
        "A. PyTorch FP32": torch_mb(fp32),
        "B. PyTorch INT8": None,  # 양자화된 Linear는 packed 가중치라 state_dict로 크기 측정이 부정확 → 생략
        "C. ORT FP32 (최적화 끔)": os.path.getsize(p_fp32) / 2**20,
        "D. ORT FP32 (최적화 켬)": os.path.getsize(p_fp32) / 2**20,
        "E. ORT INT8": os.path.getsize(p_int8) / 2**20,
        "F. ORT INT8 reduce_range": os.path.getsize(p_int8_rr) / 2**20,
        "G. ORT INT8 reduce_range+per_channel": os.path.getsize(p_int8_pc) / 2**20,
    }
    return runners, sizes


def encode(tok, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="np")
    return {k: enc[k].astype(np.int64) for k in INPUTS}


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def mcnemar_p(base_ok, new_ok):
    """같은 문장에서 한쪽만 맞힌 개수(b, c)로 정확도 차이가 우연인지 검정 (정확 이항검정, 양측)."""
    # int()로 파이썬 정수로 바꿈: numpy int64로 두면 2**n이 오버플로해서 p값이 틀림 (debugging_log #10)
    b = int(sum(bool(x) and not bool(y) for x, y in zip(base_ok, new_ok)))
    c = int(sum(bool(y) and not bool(x) for x, y in zip(base_ok, new_ok)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n * 2
    return b, c, min(p, 1.0)


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    texts, labels = load_nsmc(path, n)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    runners, sizes = make_runners(tok)

    # 1) 정확도: 2000문장을 64개씩
    probs = {}
    for name, run in runners.items():
        probs[name] = np.concatenate([softmax(run(encode(tok, texts[i:i + 64]))) for i in range(0, n, 64)])
    oks = {k: [int(p.argmax()) == y for p, y in zip(v, labels)] for k, v in probs.items()}

    ref = probs["A. PyTorch FP32"]
    onnx_diff = np.abs(probs["D. ORT FP32 (최적화 켬)"] - ref).max()

    # 2) 속도: warm-up 후 라운드마다 순서를 섞어 번갈아 측정 → 라운드 평균의 중앙값
    cpu_before = psutil.cpu_percent(interval=1.0)
    lat = {bs: {k: [] for k in runners} for bs in (1, 8, 32)}
    batches = {bs: [encode(tok, texts[i:i + bs]) for i in range(0, 32 * 20, bs)][:20] for bs in lat}
    for run in runners.values():
        for b in batches[1][:5]:
            run(b)  # warm-up
    order = list(runners)
    for _ in range(5):
        random.shuffle(order)
        for bs in lat:
            for name in order:
                s = time.perf_counter()
                for b in batches[bs]:
                    runners[name](b)
                lat[bs][name].append((time.perf_counter() - s) / len(batches[bs]) * 1000 / bs)
    cpu_after = psutil.cpu_percent(interval=1.0)

    print(f"환경: torch {torch.__version__}, onnxruntime {ort.__version__}, threads={THREADS}, "
          f"측정 전/후 CPU 사용률 {cpu_before:.0f}% / {cpu_after:.0f}%")
    print(f"PyTorch FP32 ↔ ORT FP32 확률 최대 차이: {onnx_diff:.2e}\n")

    base = statistics.median(lat[1]["A. PyTorch FP32"])
    conf = ref.max(-1) >= 0.9  # FP32가 확신(≥0.9)한 문장
    print("| 변형 | 크기(MB) | 정확도 | A 대비 McNemar (b/c, p) | A와 예측 다름 (그중 A 확신≥0.9) "
          "| 긍정확률 차이 평균/최대 | 문장당 ms (bs=1) | bs=8 | bs=32 | A 대비 배속(bs=1) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in runners:
        b, c, p = mcnemar_p(oks["A. PyTorch FP32"], oks[name])
        flip = ref.argmax(-1) != probs[name].argmax(-1)
        gap = np.abs(ref[:, 1] - probs[name][:, 1])
        m = {bs: statistics.median(lat[bs][name]) for bs in lat}
        size = f"{sizes[name]:.1f}" if sizes[name] else "-"
        print(f"| {name} | {size} | {np.mean(oks[name]):.2%} | {b}/{c}, p={p:.2f} | {flip.sum()} ({(flip & conf).sum()}) "
              f"| {gap.mean():.4f}/{gap.max():.4f} | {m[1]:.2f} | {m[8]:.2f} | {m[32]:.2f} | {base / m[1]:.2f}x |")

    # 3) FP32와 INT8 예측이 달라진 문장: 두 모델의 확률을 나란히 (미션 16 피드백)
    for final in ("E. ORT INT8", "G. ORT INT8 reduce_range+per_channel"):
        print(f"\n[{final}] 예측이 바뀐 문장 (FP32 긍정확률 → INT8 긍정확률, 정답)")
        for i, (pa, pe) in enumerate(zip(ref, probs[final])):
            if pa.argmax() != pe.argmax():
                print(f"- {texts[i][:40]!r}: {pa[1]:.2f} → {pe[1]:.2f}, 정답={'긍정' if labels[i] else '부정'}")


if __name__ == "__main__":
    main()
