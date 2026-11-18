"""감성 분석 모델 선정 실험.

NSMC 테스트셋(ratings_test.txt)에서 무작위로 뽑은 N개 문장으로 후보 모델의
정확도·크기·CPU 지연시간을 같은 조건에서 비교한다.

사용법: python select_model.py <ratings_test.txt 경로> [N]
"""
import csv
import random
import statistics
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CANDIDATES = [
    "daekeun-ml/koelectra-small-v3-nsmc",
    "monologg/koelectra-small-finetuned-nsmc",
    "monologg/koelectra-base-finetuned-nsmc",
    "cringepnh/koelectra-korean-sentiment",
]
THREADS = 4
torch.set_num_threads(THREADS)  # 서버(CPU) 환경을 가정해 스레드 수를 고정


def load_nsmc(path, n, seed=42):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["document"]]
    random.Random(seed).shuffle(rows)  # 시드 고정 → 모든 모델이 같은 문장으로 평가됨
    rows = rows[:n]
    return [r["document"] for r in rows], [int(r["label"]) for r in rows]


@torch.inference_mode()
def predict(tok, model, texts, bs=64):
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt")
        preds += model(**enc).logits.argmax(-1).tolist()
    return preds


@torch.inference_mode()
def latency_ms(tok, model, texts):
    """리뷰 1개씩(batch=1) 처리하는 실제 서비스 상황의 지연시간."""
    times = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=128, return_tensors="pt")
        s = time.perf_counter()
        model(**enc)
        times.append((time.perf_counter() - s) * 1000)
    return times


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    texts, labels = load_nsmc(path, n)
    print(f"NSMC test 샘플 {n}개 (긍정 비율 {sum(labels) / n:.1%}), torch {torch.__version__}, threads={THREADS}")

    models = {}
    for name in CANDIDATES:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        models[name] = (tok, model)

    results = {}
    for name, (tok, model) in models.items():
        preds = predict(tok, model, texts)
        acc = sum(p == y for p, y in zip(preds, labels)) / n
        params = sum(p.numel() for p in model.parameters())
        results[name] = {"acc": acc, "params_m": params / 1e6, "mb": params * 4 / 2**20,
                         "id2label": model.config.id2label, "lat": []}

    # 지연시간: warm-up 후 모델을 번갈아 3라운드 측정 → 중앙값 (미션 16·17 측정 원칙)
    lat_texts = texts[:100]
    for tok, model in models.values():
        latency_ms(tok, model, lat_texts[:10])  # warm-up
    for _ in range(3):
        for name, (tok, model) in models.items():
            results[name]["lat"] += latency_ms(tok, model, lat_texts)

    print(f"\n| 모델 | 파라미터(M) | FP32 크기(MB) | NSMC 정확도 | 지연 중앙값(ms, batch=1) | id2label |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['params_m']:.1f} | {r['mb']:.0f} | {r['acc']:.2%} | "
              f"{statistics.median(r['lat']):.1f} | {r['id2label']} |")


if __name__ == "__main__":
    main()
