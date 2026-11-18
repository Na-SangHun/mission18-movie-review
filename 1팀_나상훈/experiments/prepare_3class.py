"""3클래스(부정/중립/긍정) 학습 데이터 만들기 — NSMC 원본(raw/, 평점 1~10 포함)에서.

NSMC 기준을 따르되 NSMC가 버린 중립을 살림: 1~4 부정(0), 5~8 중립(1), 9~10 긍정(2)
누수 방지
  1. 영화 단위로 train/val/test 분리 (같은 영화 리뷰가 양쪽에 있으면 제목·배우명 암기로 점수가 부풀 수 있음)
  2. 같은 문장이 여러 평점으로 달린 경우(예: '재밌어요' 10점·7점) 라벨이 모순 → 제거
  3. test/val 문장 중 train에 똑같이 있는 문장 제거
클래스 균형: 긍정이 부정의 3배 → 클래스마다 같은 개수로 뽑음

사용법: python prepare_3class.py <nsmc-master/raw 폴더> <출력 폴더>
출력: train.tsv, val.tsv, test.tsv (열: text, rating, movie_id)
"""
import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

SEED = 42
PER_CLASS = {"train": 50000, "val": 3000, "test": 3000}


def label3(rating: int) -> int:
    return 0 if rating <= 4 else (1 if rating <= 8 else 2)


def main():
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for f in glob.glob(os.path.join(raw_dir, "*.json")):
        for x in json.load(open(f, encoding="utf-8")):
            text = " ".join(x["review"].split())  # 줄바꿈·연속 공백 정리
            if text:
                rows.append((text, int(x["rating"]), x["movie_id"]))
    print(f"원본 리뷰 {len(rows):,}개, 영화 {len({r[2] for r in rows}):,}편")

    # 2. 같은 문장이 서로 다른 클래스로 달린 경우 제거, 같은 클래스 중복은 1개만 남김
    classes = defaultdict(set)
    for t, r, _ in rows:
        classes[t].add(label3(r))
    seen, kept = set(), []
    for t, r, m in rows:
        if len(classes[t]) == 1 and t not in seen:
            seen.add(t)
            kept.append((t, r, m))
    print(f"중복·라벨 모순 제거 후 {len(kept):,}개 (모순 문장 {sum(len(v) > 1 for v in classes.values()):,}종 제거)")

    # 1. 영화 단위 분리 8:1:1
    movies = sorted({m for _, _, m in kept})
    random.Random(SEED).shuffle(movies)
    n = len(movies)
    split_of = {m: ("train" if i < n * 0.8 else "val" if i < n * 0.9 else "test") for i, m in enumerate(movies)}
    by_split = defaultdict(list)
    for row in kept:
        by_split[split_of[row[2]]].append(row)

    # 중복 제거를 전체에 대해 했으므로 3번(train과 같은 문장)은 이미 보장됨 → 확인만
    train_texts = {t for t, _, _ in by_split["train"]}
    assert not any(t in train_texts for s in ("val", "test") for t, _, _ in by_split[s])

    rng = random.Random(SEED)
    for split, k in PER_CLASS.items():
        pool = defaultdict(list)
        for row in by_split[split]:
            pool[label3(row[1])].append(row)
        print(f"{split}: 사용 가능 {dict(sorted(Counter({c: len(v) for c, v in pool.items()}).items()))}", end=" → ")
        picked = []
        for c in (0, 1, 2):
            rng.shuffle(pool[c])
            picked += pool[c][:k]
        rng.shuffle(picked)
        with open(os.path.join(out_dir, f"{split}.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["text", "rating", "movie_id"])
            w.writerows(picked)
        print(f"{len(picked):,}개 저장 (클래스당 {k:,})")


if __name__ == "__main__":
    main()
