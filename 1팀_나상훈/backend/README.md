# 영화 리뷰 감성 분석 API (코드잇 미션 18 백엔드)

- FastAPI + PostgreSQL. 영화·리뷰 CRUD, 리뷰 등록 시 감성 분석 자동 실행
- 감성 모델: NSMC 원본(평점 1~10)으로 직접 학습한 **부정/중립/긍정** 3클래스 KoELECTRA-small, ONNX INT8 동적 양자화(14MB) + temperature 보정(T=1.34)
- 평점 = 리뷰 감성 점수(긍정 확률 + 0.5 × 중립 확률)의 평균 (0~1)
- API 문서: `/docs`

## 실행
```bash
# 로컬 Postgres (테스트용)
docker run -d -p 5432:5432 -e POSTGRES_USER=app -e POSTGRES_PASSWORD=app -e POSTGRES_DB=appdb postgres:16-alpine

# 방법 1: 파이썬
pip install -r requirements.txt && uvicorn main:app --reload        # http://localhost:8000/docs

# 방법 2: Docker 이미지
docker run -p 8000:8000 -e DATABASE_URL="postgresql://..." nasanghun/movie-sentiment-api
```
`DATABASE_URL`이 없으면 `postgresql://app:app@localhost:5432/appdb`를 사용합니다.
