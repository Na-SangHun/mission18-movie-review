"""시연 데이터 등록: 영화 4편 × 리뷰 12개 (미션 요구: 영화 3개 이상, 영화당 리뷰 10개 이상).

API로 등록하므로 감성 분석도 실제 서비스와 똑같이 실행됨. 같은 제목의 영화가 이미 있으면 건너뜀.
리뷰는 직접 작성했고, 작성할 때 의도한 감성(부정/중립/긍정)을 같이 적어 모델 판정과 비교함.

사용법: python seed_data.py https://movie-sentiment-api-latest.onrender.com
"""
import sys
from collections import Counter

import requests

MOVIES = [
    ({"title": "기생충", "release_date": "2019-05-30", "director": "봉준호", "genre": "드라마, 스릴러",
      "poster_url": "https://i.namu.wiki/i/mc8hQnItqB1YF4hc3Otb3Ra7SW0pAgEBLOmP3Y-sGqYoD8rdTK6KudlbX-6DvardkV9j5_MTug_"
                    "675WY1mN0j5v-hUuABhaSoqN6vQX25Kq6F87kfsfwOTGZI-OYQzRx2gT5d6VvKoSE-MIi5ALgLFhJ9czfBCoalhL5sIC_zFw.webp"},
     [("김민준", "계단과 냄새 하나로 계급을 이렇게 보여줄 수 있다니 감탄했어요", "긍정"),
      ("이서연", "배우들 연기가 하나같이 완벽하고 끝까지 긴장을 놓을 수 없었다", "긍정"),
      ("박지호", "두 번 봤는데 두 번째가 더 소름 돋았습니다. 최고의 한국 영화", "긍정"),
      ("최유나", "웃기다가 갑자기 무서워지는 전개가 정말 대단해요", "긍정"),
      ("정하은", "칸이랑 아카데미를 다 받은 이유가 있네요. 강력 추천", "긍정"),
      ("강도윤", "상징이 많아서 생각할 거리가 많은 영화. 여운이 길게 남습니다", "긍정"),
      ("윤서준", "잘 만든 건 알겠는데 후반부는 저한테는 좀 과했어요", "중립"),
      ("임채원", "기대가 너무 커서 그런지 생각보다는 평범했어요", "중립"),
      ("한지우", "연출은 좋았지만 결말이 너무 찝찝해서 다시 보고 싶진 않네요", "중립"),
      ("오세훈", "그냥 그랬어요. 왜 이렇게 극찬받는지 잘 모르겠음", "중립"),
      ("서민지", "불편하고 우울하기만 했다. 시간 아까웠어요", "부정"),
      ("신동현", "과대평가된 영화. 억지스러운 설정이 너무 많아요", "부정")]),
    ({"title": "인터스텔라", "release_date": "2014-11-06", "director": "크리스토퍼 놀란", "genre": "SF, 드라마",
      "poster_url": "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg"},
     [("김태현", "아이맥스로 봤는데 우주 장면이 압도적이었다. 인생 영화", "긍정"),
      ("이수빈", "아빠와 딸 이야기에 펑펑 울었습니다. 음악도 최고", "긍정"),
      ("박준영", "과학적 설정이 탄탄해서 볼수록 빠져드는 영화", "긍정"),
      ("최다은", "몇 번을 다시 봐도 감동적이에요. 놀란 감독 최고작", "긍정"),
      ("정우진", "블랙홀 장면은 지금 봐도 소름. 꼭 큰 화면으로 보세요", "긍정"),
      ("강예린", "시간이 가는 줄 모르고 봤어요. 강력 추천합니다", "긍정"),
      ("윤재민", "영상은 멋있는데 중간중간 설명이 길어서 좀 지루했다", "중립"),
      ("임소희", "어려워서 반은 이해 못 했지만 볼만은 했어요", "중립"),
      ("한승우", "나쁘지 않았어요. 다만 러닝타임이 너무 길어요", "중립"),
      ("오지민", "감동은 있었는데 결말은 좀 억지스러웠던 것 같아요", "중립"),
      ("서현우", "3시간 동안 졸다 깼다 했네요. 너무 지루함", "부정"),
      ("신하영", "물리 얘기만 잔뜩 하고 정작 재미는 없었다", "부정")]),
    ({"title": "부산행", "release_date": "2016-07-20", "director": "연상호", "genre": "액션, 스릴러",
      "poster_url": "https://upload.wikimedia.org/wikipedia/en/9/95/Train_to_Busan.jpg"},
     [("김도현", "한국 좀비 영화가 이렇게 재밌을 줄 몰랐어요. 긴장감 최고", "긍정"),
      ("이지은", "공유 연기에 마지막에 눈물 났습니다", "긍정"),
      ("박성민", "기차라는 좁은 공간을 정말 잘 활용했다. 몰입감 대박", "긍정"),
      ("최윤아", "마동석 배우 나올 때마다 속이 시원했어요", "긍정"),
      ("정민호", "처음부터 끝까지 쉴 틈이 없는 영화. 강추", "긍정"),
      ("강수아", "킬링타임용으로 무난했어요", "중립"),
      ("윤현준", "초반은 재밌었는데 후반 신파가 좀 길었어요", "중립"),
      ("임가은", "좀비는 무서웠는데 이야기는 뻔했다", "중립"),
      ("한예준", "그럭저럭 볼만했어요. 특별히 기억에 남지는 않음", "중립"),
      ("오나연", "이기적인 캐릭터 때문에 보는 내내 답답하고 화났어요", "부정"),
      ("서지훈", "억지 신파 때문에 다 망쳤다. 실망", "부정"),
      ("신유진", "개연성이 너무 없어서 집중이 안 됐어요. 별로", "부정")]),
    ({"title": "다크 나이트", "release_date": "2008-08-06", "director": "크리스토퍼 놀란", "genre": "액션, 범죄",
      "poster_url": "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg"},
     [("김현수", "히스 레저의 조커는 영화 역사에 남을 연기다", "긍정"),
      ("이민서", "히어로 영화가 이렇게 묵직할 수 있다니. 최고의 걸작", "긍정"),
      ("박지원", "볼 때마다 새로운 게 보이는 영화. 각본이 완벽해요", "긍정"),
      ("최준혁", "긴장감이 처음부터 끝까지 유지되는 명작입니다", "긍정"),
      ("정소윤", "조커와 배트맨의 대립이 정말 소름 돋아요", "긍정"),
      ("강민재", "음악, 연출, 연기 모두 완벽. 꼭 보세요", "긍정"),
      ("윤하린", "잘 만들었지만 너무 어둡고 무거웠어요", "중립"),
      ("임도윤", "조커는 좋았는데 나머지는 평범했다", "중립"),
      ("한서윤", "명작이라길래 봤는데 저한테는 그냥 그랬어요", "중립"),
      ("오준서", "후반부 투페이스 이야기는 좀 급하게 끝난 느낌", "중립"),
      ("서은우", "너무 길고 복잡해서 보다가 지쳤어요", "부정"),
      ("신지아", "액션도 별로고 대사만 많아서 지루했다", "부정")]),
]
LABELS = ["부정", "중립", "긍정"]


def main():
    api = sys.argv[1].rstrip("/")
    existing = {m["title"] for m in requests.get(f"{api}/movies", timeout=120).json()}
    rows = []
    for movie, reviews in MOVIES:
        if movie["title"] in existing:
            print(f"건너뜀 (이미 있음): {movie['title']}")
            continue
        r = requests.post(f"{api}/movies", json=movie, timeout=60)
        r.raise_for_status()
        mid = r.json()["id"]
        for author, content, intended in reviews:
            rv = requests.post(f"{api}/reviews", json={"movie_id": mid, "author": author, "content": content},
                               timeout=60)
            rv.raise_for_status()
            rows.append((movie["title"], content, intended, rv.json()))
        rating = requests.get(f"{api}/movies/{mid}/rating", timeout=60).json()
        print(f"{movie['title']}: 리뷰 {rating['review_count']}개, 평점 {rating['rating'] * 10:.1f} / 10")

    if not rows:
        return
    print("\n| 영화 | 리뷰 | 작성 의도 | 모델 판정 | 부정 / 중립 / 긍정 |")
    print("|---|---|---|---|---|")
    for title, content, intended, r in rows:
        mark = "" if r["sentiment"] == intended else " ⚠"
        print(f"| {title} | {content} | {intended} | {r['sentiment']}{mark} | "
              f"{r['prob_neg']:.2f} / {r['prob_neu']:.2f} / {r['prob_pos']:.2f} |")
    agree = sum(r["sentiment"] == i for _, _, i, r in rows)
    print(f"\n작성 의도와 일치: {agree} / {len(rows)}")
    conf = Counter((i, r["sentiment"]) for _, _, i, r in rows)
    print("| 의도＼판정 | 부정 | 중립 | 긍정 |\n|---|---|---|---|")
    for i in LABELS:
        print(f"| {i} | " + " | ".join(str(conf[(i, p)]) for p in LABELS) + " |")


if __name__ == "__main__":
    main()
