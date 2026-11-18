# 모델 선정 실험 결과 (select_model.py 실행 결과)

- 일시: 2026-11-18, Windows 11, Python 3.14.6, torch 2.13.0+cpu, transformers 5.23.0
- CPU 12 논리 코어 중 `torch.set_num_threads(4)`로 4개 사용, 다른 작업을 돌리지 않은 상태
- 데이터: NSMC `ratings_test.txt`에서 시드 42로 무작위 추출한 2,000개 (긍정 52.9%)
- 지연시간: 리뷰 1개(batch=1), warm-up 후 4개 모델을 번갈아 3라운드 × 100문장 측정, 중앙값

| 모델 | 파라미터(M) | FP32 크기(MB) | NSMC 정확도 | 지연 중앙값(ms) | id2label |
|---|---|---|---|---|---|
| daekeun-ml/koelectra-small-v3-nsmc | 14.1 | 54 | 89.05% | 10.1 | {0: '0', 1: '1'} |
| monologg/koelectra-small-finetuned-nsmc | 13.8 | 53 | 87.65% | 9.8 | {0: 'negative', 1: 'positive'} |
| monologg/koelectra-base-finetuned-nsmc | 110.8 | 423 | 89.70% | 40.1 | {0: 'negative', 1: 'positive'} |
| cringepnh/koelectra-korean-sentiment | 112.9 | 431 | 89.60% | 42.1 | {0: 'negative', 1: 'positive'} |

정확도 95% 신뢰구간: n=2000, p≈0.89일 때 ±1.4%p (√(p(1-p)/n)×1.96)
