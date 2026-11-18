# 영화 리뷰 감성 분석 API (FastAPI + ONNX INT8) — Hugging Face Spaces(Docker)용
FROM python:3.12-slim

# HF Spaces는 컨테이너를 uid 1000 사용자로 실행 → 그 사용자 소유 폴더에서 실행해야 SQLite 파일을 쓸 수 있음
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# 의존성 레이어를 먼저: 코드만 바뀌면 pip install 단계는 캐시 재사용
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user main.py sentiment.py ./
COPY --chown=user model/ko3_int8.onnx model/tokenizer.json ./model/

# HF Spaces 기본 포트 7860
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
