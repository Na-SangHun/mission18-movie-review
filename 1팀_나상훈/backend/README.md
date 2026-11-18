---
title: Movie Review Sentiment API
emoji: 🎬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 영화 리뷰 감성 분석 API (코드잇 미션 18 백엔드)

- FastAPI + SQLite. 영화·리뷰 CRUD, 리뷰 등록 시 감성 분석 자동 실행
- 감성 모델: NSMC 원본(평점 1~10)으로 직접 학습한 **부정/중립/긍정** 3클래스 KoELECTRA-small, ONNX INT8 동적 양자화(14MB) + temperature 보정(T=1.34)
- 평점 = 리뷰 감성 점수(긍정 확률 + 0.5 × 중립 확률)의 평균 (0~1)
- API 문서: `/docs`

로컬 실행: `pip install -r requirements.txt && uvicorn main:app --reload` → http://localhost:8000/docs

주의: 무료 Space는 재시작하면 SQLite 데이터가 초기화됩니다.
