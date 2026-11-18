"""리뷰 감성 분석: 부정/중립/긍정 3클래스 (직접 학습한 KoELECTRA-small, ONNX INT8 동적 양자화).

왜 3클래스인가: 처음 쓴 NSMC 2클래스 모델은 중립 리뷰(평점 5~8)를 빼고 학습돼서
"배우는 좋은데 스토리가 별로"를 부정 99.7%로 판정함 → NSMC 원본(평점 포함)으로 중립까지 학습.
근거: ../experiments/eval_3class_results.md, lighten_3class_results.md

서버에는 torch/transformers 없이 tokenizers + onnxruntime만 사용 (토큰 일치 9,000/9,000 확인).
"""
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))
MAX_LEN = 128
LABELS = ["부정", "중립", "긍정"]  # 학습 때 id2label: 0=부정(평점 1~4), 1=중립(5~8), 2=긍정(9~10)
# 확률 보정(temperature scaling): val 9,000개로 NLL 최소 T → test ECE 0.053→0.009. 예측은 바뀌지 않음
TEMPERATURE = 1.34


class SentimentModel:
    def __init__(self, model_dir=MODEL_DIR, threads=2):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)  # transformers의 truncation=True, max_length=128과 같음
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads  # 무료 서버 CPU가 적어서 스레드를 작게 고정
        self.sess = ort.InferenceSession(os.path.join(model_dir, "ko3_int8.onnx"), so,
                                         providers=["CPUExecutionProvider"])

    def predict(self, text: str) -> dict:
        """리뷰 1개 → {'label', 'prob_neg', 'prob_neu', 'prob_pos'}.

        항상 1개씩(bs=1): INT8 동적 양자화는 묶음 전체로 scale을 계산해서, 묶으면 같은 문장도
        결과가 달라질 수 있음 (experiments/investigate_flips.md H1)
        """
        enc = self.tok.encode(text)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        z = self.sess.run(["logits"], feed)[0][0] / TEMPERATURE
        p = np.exp(z - z.max())
        p = p / p.sum()
        return {"label": LABELS[int(p.argmax())],
                "prob_neg": round(float(p[0]), 4), "prob_neu": round(float(p[1]), 4), "prob_pos": round(float(p[2]), 4)}
