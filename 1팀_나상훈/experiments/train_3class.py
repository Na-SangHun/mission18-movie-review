"""KoELECTRA-small을 3클래스(부정/중립/긍정)로 파인튜닝 (GPU, conda part4 환경).

시작 모델: monologg/koelectra-small-v3-discriminator — 감성으로 학습되기 '전' 사전학습 모델.
(NSMC로 이미 학습된 모델에서 시작하면 그 모델이 본 리뷰가 우리 test에 섞여 평가가 부풀 수 있음)

사용법: python train_3class.py <data3 폴더> <출력 폴더>
출력: <출력 폴더>/ (HF 형식 모델), test_preds.tsv (text, rating, label, p_neg, p_neu, p_pos)
"""
import csv
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE = "monologg/koelectra-small-v3-discriminator"
LABELS = ["부정", "중립", "긍정"]
EPOCHS, BS, LR, MAX_LEN, SEED = 3, 64, 1e-4, 128, 42  # small 모델은 base보다 큰 lr(1e-4)을 많이 씀


def label3(rating: int) -> int:
    return 0 if rating <= 4 else (1 if rating <= 8 else 2)


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r["text"] for r in rows], [int(r["rating"]) for r in rows]


@torch.inference_mode()
def predict(model, tok, texts, device):
    model.eval()
    out = []
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i + 256], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.float().cpu())
    return torch.cat(out).numpy()


def macro_f1(y, pred):
    f1s = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum()
        p = tp / max((pred == c).sum(), 1)
        r = tp / max((y == c).sum(), 1)
        f1s.append(2 * p * r / max(p + r, 1e-9))
    return float(np.mean(f1s))


def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    tr_x, tr_r = load(os.path.join(data_dir, "train.tsv"))
    va_x, va_r = load(os.path.join(data_dir, "val.tsv"))
    te_x, te_r = load(os.path.join(data_dir, "test.tsv"))
    tr_y = [label3(r) for r in tr_r]
    va_y = np.array([label3(r) for r in va_r])

    def collate(batch):
        texts, ys = zip(*batch)
        enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(ys)
        return enc

    loader = DataLoader(list(zip(tr_x, tr_y)), batch_size=BS, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)  # 처음 6%는 lr을 천천히 올림
    scaler = torch.amp.GradScaler()  # fp16 혼합 정밀도: 속도↑ 메모리↓

    best_f1 = -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        pred = predict(model, tok, va_x, device).argmax(1)
        f1, acc = macro_f1(va_y, pred), (pred == va_y).mean()
        print(f"epoch {ep}: train loss {total / len(loader):.4f}, val acc {acc:.2%}, val macro-F1 {f1:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if f1 > best_f1:  # 검증 성능이 가장 좋은 epoch만 저장 (test는 마지막에 한 번만 봄)
            best_f1 = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            print(f"  → 저장 (best val macro-F1 {f1:.4f})", flush=True)

    best = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    logits = predict(best, tok, te_x, device)
    p = torch.softmax(torch.tensor(logits), 1).numpy()
    with open(os.path.join(out_dir, "test_preds.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["text", "rating", "label", "logit_neg", "logit_neu", "logit_pos"])
        for t, r, lg in zip(te_x, te_r, logits):
            w.writerow([t, r, label3(r), *[f"{v:.5f}" for v in lg]])
    te_y = np.array([label3(r) for r in te_r])
    print(f"test acc {(p.argmax(1) == te_y).mean():.2%}, macro-F1 {macro_f1(te_y, p.argmax(1)):.4f}")


if __name__ == "__main__":
    main()
