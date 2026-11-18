"""백엔드 전체 흐름 점검. 실행: python test_main.py

로컬 테스트용 Postgres가 필요: docker run -d -p 5432:5432 -e POSTGRES_USER=app -e POSTGRES_PASSWORD=app \
    -e POSTGRES_DB=appdb postgres:16-alpine
테이블을 지우고 시작하므로 localhost DB에서만 실행되게 막음 (배포 DB를 실수로 지우지 않도록)
"""
import psycopg
from fastapi.testclient import TestClient

from main import DATABASE_URL, app

assert "@localhost" in DATABASE_URL or "@127.0.0.1" in DATABASE_URL, "로컬 DB에서만 실행하세요"
with psycopg.connect(DATABASE_URL) as conn:
    conn.execute("DROP TABLE IF EXISTS reviews, movies")

MOVIE = {"title": "기생충", "release_date": "2019-05-30", "director": "봉준호",
         "genre": "드라마", "poster_url": "https://example.com/p.jpg"}

with TestClient(app) as c:  # with 블록 → lifespan 실행(테이블 생성·모델 로드)
    m = c.post("/movies", json=MOVIE).json()
    assert m["review_count"] == 0 and m["rating"] is None

    post = lambda text: c.post("/reviews", json={"movie_id": m["id"], "author": "a", "content": text}).json()
    pos, neu, neg = post("최고의 영화, 정말 감동적이에요"), post("배우는 좋은데 스토리가 별로"), post("시간이 아까운 최악의 영화")
    assert [x["sentiment"] for x in (pos, neu, neg)] == ["긍정", "중립", "부정"], (pos, neu, neg)
    for x in (pos, neu, neg):
        assert abs(x["prob_neg"] + x["prob_neu"] + x["prob_pos"] - 1) < 1e-3  # 확률 합 = 1
        assert abs(x["score"] - (x["prob_pos"] + 0.5 * x["prob_neu"])) < 1e-3  # DB 생성 열 = 긍정 + 0.5·중립

    r = c.get(f"/movies/{m['id']}/rating").json()
    assert r["review_count"] == 3
    assert abs(r["rating"] - (pos["score"] + neu["score"] + neg["score"]) / 3) < 1e-3  # 평점 = 감성 점수 평균

    assert [x["id"] for x in c.get("/reviews?limit=1").json()] == [neg["id"]]  # 최신순
    assert len(c.get(f"/movies/{m['id']}/reviews").json()) == 3

    # 검증: 없는 영화에 리뷰 → 404, 빈 내용·잘못된 URL → 422
    assert c.post("/reviews", json={"movie_id": 999, "author": "a", "content": "x"}).status_code == 404
    assert c.post("/reviews", json={"movie_id": m["id"], "author": "a", "content": ""}).status_code == 422
    assert c.post("/movies", json=MOVIE | {"poster_url": "not-a-url"}).status_code == 422

    assert c.delete(f"/reviews/{pos['id']}").status_code == 204
    assert c.delete(f"/movies/{m['id']}").status_code == 204
    assert c.get(f"/movies/{m['id']}").status_code == 404
    assert c.get("/reviews").json() == []  # 영화 삭제 시 리뷰도 함께 삭제 (CASCADE)

print("모든 점검 통과")
