"""영화 리뷰 감성 분석 서비스 - 백엔드 (FastAPI + SQLite).

실행: uvicorn main:app --reload   →   API 문서: http://localhost:8000/docs
"""
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, HttpUrl

from sentiment import SentimentModel

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "movies.db"))
KST = timezone(timedelta(hours=9))

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL,
    release_date TEXT NOT NULL,
    director     TEXT NOT NULL,
    genre        TEXT NOT NULL,
    poster_url   TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id      INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    author        TEXT NOT NULL,
    content       TEXT NOT NULL,
    sentiment     TEXT NOT NULL CHECK (sentiment IN ('부정', '중립', '긍정')),
    prob_neg      REAL NOT NULL CHECK (prob_neg BETWEEN 0 AND 1),
    prob_neu      REAL NOT NULL CHECK (prob_neu BETWEEN 0 AND 1),
    prob_pos      REAL NOT NULL CHECK (prob_pos BETWEEN 0 AND 1),
    -- 감성 점수: 부정 0, 중립 0.5, 긍정 1의 기댓값. DB가 계산하는 생성 열 → 계산식이 한 곳에만 존재
    score         REAL GENERATED ALWAYS AS (ROUND(prob_pos + 0.5 * prob_neu, 4)) VIRTUAL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_movie ON reviews(movie_id);
"""


@contextmanager
def db():
    """요청마다 연결을 새로 열고 닫는다. SQLite는 파일 하나라서 연결 비용이 거의 없음."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row             # row["title"]처럼 이름으로 꺼내기
    conn.execute("PRAGMA foreign_keys = ON")   # SQLite는 기본값이 OFF → 켜야 CASCADE 삭제가 동작
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 켜질 때 한 번만: 테이블 생성 + 모델 로드 (요청마다 로드하면 매번 수백 ms 낭비)."""
    with db() as conn:
        conn.executescript(SCHEMA)
    app.state.model = SentimentModel()
    yield


app = FastAPI(
    title="영화 리뷰 감성 분석 API",
    description="영화 정보와 사용자 리뷰를 관리하고, 리뷰가 등록되면 KoELECTRA-small(ONNX INT8) 모델로 "
                "감성(부정/중립/긍정)을 자동 분석합니다. 리뷰의 감성 점수 = 긍정 확률 + 0.5 × 중립 확률 (0~1)이고, "
                "영화 평점은 리뷰 감성 점수의 평균입니다.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- 스키마 (Pydantic): 요청 검증 + Swagger 문서 자동 생성 ----------
class MovieCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100, description="영화 제목", examples=["기생충"])
    release_date: date = Field(description="개봉일 (YYYY-MM-DD)", examples=["2019-05-30"])
    director: str = Field(min_length=1, max_length=50, description="감독", examples=["봉준호"])
    genre: str = Field(min_length=1, max_length=50, description="장르", examples=["드라마"])
    poster_url: HttpUrl = Field(description="포스터 이미지 URL", examples=["https://example.com/poster.jpg"])


class Movie(MovieCreate):
    id: int = Field(description="영화 ID")
    created_at: datetime = Field(description="등록 시각 (KST)")
    review_count: int = Field(description="리뷰 수")
    rating: float | None = Field(description="평점: 리뷰 감성 점수의 평균 (0~1, 리뷰가 없으면 null)")


class ReviewCreate(BaseModel):
    movie_id: int = Field(description="리뷰를 남길 영화 ID", examples=[1])
    author: str = Field(min_length=1, max_length=30, description="작성자 이름", examples=["홍길동"])
    content: str = Field(min_length=1, max_length=1000, description="리뷰 내용",
                         examples=["배우들 연기가 정말 좋았고 마지막 장면이 오래 기억에 남아요"])


class Review(ReviewCreate):
    id: int = Field(description="리뷰 ID")
    sentiment: str = Field(description="감성 분석 결과 ('부정', '중립', '긍정' 중 확률이 가장 큰 것)")
    prob_neg: float = Field(description="부정 확률 (0~1, 보정됨)")
    prob_neu: float = Field(description="중립 확률 (0~1, 보정됨)")
    prob_pos: float = Field(description="긍정 확률 (0~1, 보정됨). 세 확률의 합 = 1")
    score: float = Field(description="감성 점수 = 긍정 확률 + 0.5 × 중립 확률 (0=부정, 0.5=중립, 1=긍정)")
    created_at: datetime = Field(description="등록 시각 (KST)")


class Rating(BaseModel):
    movie_id: int
    review_count: int = Field(description="평점 계산에 쓰인 리뷰 수")
    rating: float | None = Field(description="리뷰 감성 점수의 평균 (0~1, 리뷰가 없으면 null)")


# ---------- 공통 SQL ----------
MOVIE_SELECT = """
SELECT m.*, COUNT(r.id) AS review_count, ROUND(AVG(r.score), 4) AS rating
FROM movies m LEFT JOIN reviews r ON r.movie_id = m.id
"""  # 평점은 저장하지 않고 조회할 때 계산 → 리뷰가 추가·삭제돼도 값이 어긋나지 않음


def get_movie_or_404(conn, movie_id: int) -> dict:
    row = conn.execute(MOVIE_SELECT + " WHERE m.id = ? GROUP BY m.id", (movie_id,)).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"영화 {movie_id}번이 없습니다")
    return dict(row)


NOT_FOUND = {404: {"description": "해당 ID가 없음"}}


# ---------- 영화 ----------
@app.post("/movies", response_model=Movie, status_code=201, tags=["영화"], summary="영화 등록")
def create_movie(movie: MovieCreate):
    """제목·개봉일·감독·장르·포스터 URL로 영화를 등록합니다."""
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO movies (title, release_date, director, genre, poster_url, created_at) VALUES (?,?,?,?,?,?)",
            (movie.title, movie.release_date.isoformat(), movie.director, movie.genre,
             str(movie.poster_url), datetime.now(KST).isoformat(timespec="seconds")))
        return get_movie_or_404(conn, cur.lastrowid)


@app.get("/movies", response_model=list[Movie], tags=["영화"], summary="전체 영화 조회")
def list_movies():
    """등록된 모든 영화를 최근 등록 순으로 반환합니다. 각 영화의 리뷰 수와 평점이 함께 포함됩니다."""
    with db() as conn:
        return [dict(r) for r in conn.execute(MOVIE_SELECT + " GROUP BY m.id ORDER BY m.id DESC")]


@app.get("/movies/{movie_id}", response_model=Movie, tags=["영화"], summary="특정 영화 조회", responses=NOT_FOUND)
def get_movie(movie_id: int):
    with db() as conn:
        return get_movie_or_404(conn, movie_id)


@app.delete("/movies/{movie_id}", status_code=204, tags=["영화"], summary="특정 영화 삭제", responses=NOT_FOUND)
def delete_movie(movie_id: int):
    """영화를 삭제합니다. 이 영화의 리뷰도 함께 삭제됩니다 (ON DELETE CASCADE)."""
    with db() as conn:
        if conn.execute("DELETE FROM movies WHERE id = ?", (movie_id,)).rowcount == 0:
            raise HTTPException(404, f"영화 {movie_id}번이 없습니다")


@app.get("/movies/{movie_id}/rating", response_model=Rating, tags=["영화"], summary="영화 평점 조회",
         responses=NOT_FOUND)
def get_rating(movie_id: int):
    """평점 = 이 영화 리뷰들의 감성 점수(긍정 확률 + 0.5 × 중립 확률) 평균. 0에 가까울수록 부정, 1에 가까울수록 긍정."""
    with db() as conn:
        m = get_movie_or_404(conn, movie_id)
    return {"movie_id": movie_id, "review_count": m["review_count"], "rating": m["rating"]}


# ---------- 리뷰 ----------
@app.post("/reviews", response_model=Review, status_code=201, tags=["리뷰"], summary="리뷰 등록 (감성 분석 자동 실행)",
          responses=NOT_FOUND)
def create_review(review: ReviewCreate):
    """리뷰를 저장하기 전에 감성 분석 모델을 실행하고, 결과(부정/중립/긍정과 각 확률, 감성 점수)를 함께 저장해 반환합니다."""
    with db() as conn:
        get_movie_or_404(conn, review.movie_id)  # 없는 영화에 리뷰를 다는 것을 막음
        result = app.state.model.predict(review.content)
        cur = conn.execute(
            "INSERT INTO reviews (movie_id, author, content, sentiment, prob_neg, prob_neu, prob_pos, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (review.movie_id, review.author, review.content, result["label"],
             result["prob_neg"], result["prob_neu"], result["prob_pos"],
             datetime.now(KST).isoformat(timespec="seconds")))
        return dict(conn.execute("SELECT * FROM reviews WHERE id = ?", (cur.lastrowid,)).fetchone())


@app.get("/reviews", response_model=list[Review], tags=["리뷰"], summary="전체 리뷰 조회 (최신순)")
def list_reviews(limit: int = Query(10, ge=1, le=100, description="가져올 리뷰 수 (최신순)")):
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM reviews ORDER BY id DESC LIMIT ?", (limit,))]


@app.get("/movies/{movie_id}/reviews", response_model=list[Review], tags=["리뷰"], summary="특정 영화의 리뷰 조회",
         responses=NOT_FOUND)
def list_movie_reviews(movie_id: int):
    with db() as conn:
        get_movie_or_404(conn, movie_id)
        return [dict(r) for r in conn.execute(
            "SELECT * FROM reviews WHERE movie_id = ? ORDER BY id DESC", (movie_id,))]


@app.delete("/reviews/{review_id}", status_code=204, tags=["리뷰"], summary="리뷰 삭제", responses=NOT_FOUND)
def delete_review(review_id: int):
    with db() as conn:
        if conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,)).rowcount == 0:
            raise HTTPException(404, f"리뷰 {review_id}번이 없습니다")
