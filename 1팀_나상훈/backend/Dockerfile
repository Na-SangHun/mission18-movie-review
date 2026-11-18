# 영화 리뷰 감성 분석 API (FastAPI + ONNX INT8) — Render(Docker Hub 이미지)용
# DB는 컨테이너 밖 PostgreSQL(Neon): 실행할 때 DATABASE_URL 환경변수로 주소를 넘김
FROM python:3.12-slim

# root가 아닌 일반 사용자로 실행 (컨테이너가 뚫려도 피해를 줄이는 기본 습관)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# 의존성 레이어를 먼저: 코드만 바뀌면 pip install 단계는 캐시 재사용
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user main.py sentiment.py ./
COPY --chown=user model/ko3_int8.onnx model/tokenizer.json ./model/

# Render는 PORT 환경변수로 포트를 알려 줌 (없으면 8000)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
