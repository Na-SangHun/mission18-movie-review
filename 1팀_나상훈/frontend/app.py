"""영화 리뷰 감성 분석 서비스 - 프론트엔드 (Streamlit).

화면만 담당하고, 모든 데이터는 백엔드(FastAPI)에 HTTP로 요청한다. (Streamlit 안에는 저장하지 않음)
실행: streamlit run app.py   (백엔드 주소: 환경변수 또는 .streamlit/secrets.toml의 BACKEND_URL)
"""
import html
import os
from datetime import date

import requests
import streamlit as st

st.set_page_config(page_title="영화 리뷰 감성 분석", page_icon=":material/movie:", layout="wide")


def backend_url() -> str:
    """배포(Streamlit Cloud)에서는 Secrets, 로컬에서는 환경변수나 기본값을 사용."""
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"].rstrip("/")
    except FileNotFoundError:  # secrets.toml이 없는 로컬 환경
        pass
    return os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


API = backend_url()
FIELD_NAMES = {"title": "제목", "release_date": "개봉일", "director": "감독", "genre": "장르",
               "poster_url": "포스터 URL", "author": "작성자 이름", "content": "리뷰 내용"}


def api(method: str, path: str, **kwargs):
    """백엔드 호출 공통 함수: 실패하면 화면에 이유를 보여 주고 None 반환."""
    try:
        r = requests.request(method, API + path, timeout=60, **kwargs)  # 무료 서버가 잠에서 깰 때를 고려해 넉넉히
    except requests.RequestException as e:
        st.error(f"백엔드({API})에 연결할 수 없습니다: {type(e).__name__}")
        return None
    if r.status_code >= 400:
        is_json = r.headers.get("content-type", "").startswith("application/json")
        detail = r.json().get("detail", r.text) if is_json else r.text
        if isinstance(detail, list):  # Pydantic 검증 오류 → "필드: 이유" 형태로 정리
            detail = ", ".join(f"{FIELD_NAMES.get(d['loc'][-1], d['loc'][-1])}: {d['msg']}" for d in detail)
        st.error(f"요청 실패 ({r.status_code}): {detail}")
        return None
    return r.json() if r.content else True


def flash(msg: str):
    """st.rerun() 뒤에도 보이도록 알림을 세션에 잠깐 저장 (데이터 저장이 아니라 알림용)."""
    st.session_state["flash"] = msg
    st.rerun()


def poster(url: str):
    """포스터를 2:3 비율 틀에 맞춰 표시 → 원본 비율이 달라도 카드 크기가 같아짐 (넘치는 부분은 잘림).
    st.image는 높이·자르기를 지원하지 않아서 HTML img를 사용. URL은 사용자 입력이므로 escape해서
    따옴표로 속성을 탈출하는 HTML 주입을 막음 (백엔드도 http/https URL만 허용)."""
    st.markdown(f'<img src="{html.escape(url, quote=True)}" alt="포스터" style="width:100%; aspect-ratio:2/3; '
                f'object-fit:cover; border-radius:0.5rem;">', unsafe_allow_html=True)


BADGE = {"긍정": "green", "중립": "gray", "부정": "red"}


def sentiment_text(review: dict) -> str:
    """결과와 확신도를 함께 표시 (미션 16 피드백: 클래스만 보여 주지 않기)."""
    return (f":{BADGE[review['sentiment']]}-badge[{review['sentiment']}] "
            f"부정 {review['prob_neg']:.0%} · 중립 {review['prob_neu']:.0%} · 긍정 {review['prob_pos']:.0%}")


if msg := st.session_state.pop("flash", None):
    st.toast(msg, icon=":material/check_circle:")

st.title("영화 리뷰 감성 분석")
st.caption(f"리뷰를 남기면 KoELECTRA-small(ONNX INT8) 모델이 부정·중립·긍정을 자동 분석합니다 · 백엔드 API 문서: {API}/docs")

movies = api("GET", "/movies")
if movies is None:
    st.stop()
titles = {m["id"]: m["title"] for m in movies}

tab_movies, tab_add, tab_reviews = st.tabs(
    [":material/movie: 영화 목록", ":material/add: 영화 추가", ":material/rate_review: 리뷰"])


@st.dialog("영화 삭제")
def confirm_delete(movie):
    st.write(f"**{movie['title']}**을(를) 삭제할까요? 이 영화의 리뷰 {movie['review_count']}개도 함께 삭제됩니다.")
    if st.button("삭제", type="primary"):
        if api("DELETE", f"/movies/{movie['id']}"):
            flash(f"'{movie['title']}' 삭제 완료")


# ---------------- 영화 목록 ----------------
with tab_movies:
    if not movies:
        st.info("등록된 영화가 없습니다. '영화 추가' 탭에서 등록해 주세요.")
    for row in range(0, len(movies), 4):  # 4개씩 한 줄: 같은 줄 카드끼리 높이를 맞추려면 줄마다 columns를 새로 만듦
        for col, m in zip(st.columns(4), movies[row:row + 4]):
            with col.container(border=True, height="stretch"):  # stretch: 같은 줄에서 가장 긴 카드에 높이를 맞춤
                poster(m["poster_url"])
                st.subheader(m["title"])
                st.caption(f"{m['release_date']} 개봉 · {m['director']} · {m['genre']}")
                if m["rating"] is None:
                    st.markdown("평점 없음 (리뷰 0개)")
                else:  # 평점 = 리뷰 감성 점수(긍정 + 0.5·중립)의 평균 → 10점 만점으로 환산해 표시
                    st.markdown(f":material/star: **{m['rating'] * 10:.1f}** / 10 · 리뷰 {m['review_count']}개")
                if st.button("삭제", key=f"del{m['id']}", icon=":material/delete:"):
                    confirm_delete(m)

# ---------------- 영화 추가 ----------------
with tab_add:
    with st.form("add_movie", clear_on_submit=True):  # form: 버튼을 누를 때만 한 번 실행 (입력할 때마다 rerun 방지)
        title = st.text_input("제목")
        c1, c2, c3 = st.columns(3)
        release = c1.date_input("개봉일", value=date.today(), min_value=date(1900, 1, 1))
        director = c2.text_input("감독")
        genre = c3.text_input("장르")
        poster = st.text_input("포스터 URL", placeholder="https://...")
        if st.form_submit_button("영화 등록", type="primary"):
            body = {"title": title, "release_date": release.isoformat(), "director": director,
                    "genre": genre, "poster_url": poster}
            if api("POST", "/movies", json=body):
                flash(f"'{title}' 등록 완료")

# ---------------- 리뷰 ----------------
with tab_reviews:
    st.subheader("리뷰 등록")
    if not movies:
        st.info("먼저 영화를 등록해 주세요.")
    else:
        with st.form("add_review", clear_on_submit=True):
            movie_id = st.selectbox("영화", list(titles), format_func=lambda i: f"{titles[i]} (ID {i})")
            author = st.text_input("작성자 이름")
            content = st.text_area("리뷰 내용")
            submitted = st.form_submit_button("등록하고 감성 분석", type="primary")
        if submitted:  # 등록 버튼 → 백엔드가 저장 전에 감성 분석을 자동 실행하고 결과를 돌려줌
            with st.spinner("감성 분석 중..."):
                review = api("POST", "/reviews", json={"movie_id": movie_id, "author": author, "content": content})
            if review:
                # 영화 목록(평점·리뷰 수)은 스크립트 맨 위에서 이미 가져왔으므로, 다시 실행해야 새 리뷰가 반영됨
                st.session_state["last_review"] = review
                st.rerun()
        if review := st.session_state.pop("last_review", None):  # rerun 뒤 한 번만 결과 표시
            st.success("리뷰가 등록되었습니다.")
            st.markdown(f"**감성 분석 결과:** {sentiment_text(review)}")
            st.progress(review["score"], text=f"감성 점수 {review['score'] * 10:.1f} / 10 (부정 0 · 중립 5 · 긍정 10)")

    st.subheader("리뷰 목록")
    scope = st.selectbox("보기", [None, *titles], format_func=lambda i: "최근 리뷰 10개 (전체 영화)" if i is None
                         else f"{titles[i]}의 모든 리뷰")
    reviews = api("GET", "/reviews", params={"limit": 10}) if scope is None else api("GET", f"/movies/{scope}/reviews")
    if reviews:
        st.dataframe(
            [{"영화 ID": r["movie_id"], "영화": titles.get(r["movie_id"], "-"),
              "등록일": r["created_at"][:16].replace("T", " "), "작성자": r["author"], "리뷰 내용": r["content"],
              "감성": r["sentiment"], "부정·중립·긍정": f"{r['prob_neg']:.0%} · {r['prob_neu']:.0%} · {r['prob_pos']:.0%}",
              "감성 점수": r["score"] * 10} for r in reviews],
            column_config={"감성 점수": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=10),
                           "리뷰 내용": st.column_config.TextColumn(width="large")},
            hide_index=True, height="content")  # 기본값 "auto"는 10행까지만 보이고 스크롤 → 영화별 보기에서 전부 표시
    elif reviews == []:
        st.caption("아직 리뷰가 없습니다.")
