from statistics import mean
from typing import Dict, List

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score

# CIDEr (COCO caption eval)
try:
    from pycocoevalcap.cider.cider import Cider
    _CIDER_OK = True
except Exception:
    Cider = None
    _CIDER_OK = False


class BLUE:
    def __init__(self, ngrams: int = 4) -> None:
        self.smoothing = SmoothingFunction().method3
        self.n = ngrams
        weights = [1 / ngrams if i <= ngrams else 0 for i in range(1, 5)]
        self.weights = tuple(weights)

    def __call__(self, references, hypothesis) -> float:
        score = sentence_bleu(
            references,
            hypothesis,
            weights=self.weights,
            smoothing_function=self.smoothing
        )
        return score

    def __repr__(self) -> str:
        return f"bleu{self.n}"


class GLEU:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return sentence_gleu(*args, **kwargs)

    def __repr__(self):
        return "gleu"


class METEOR:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return meteor_score(*args, **kwargs)

    def __repr__(self):
        return "meteor"


class CIDEr:
    """
    CIDEr expects:
      gts  : {id: [ref_sentence_str, ...]}
      res  : {id: [hyp_sentence_str]}
    and returns (score, scores_per_sample)
    """
    def __init__(self) -> None:
        if not _CIDER_OK:
            raise ImportError(
                "pycocoevalcap is not installed or cannot be imported. "
                "Install it first, e.g. pip install pycocoevalcap"
            )
        self.scorer = Cider()

    @staticmethod
    def _join_tokens(tokens: List[str]) -> str:
        return " ".join(tokens).strip()

    def __call__(self,
                 refs_tokens: List[List[str]],
                 hypo_tokens: List[str]) -> float:
        # 单样本接口：把它包装成 COCO 所需 dict，再取该样本分数
        gts = {0: [self._join_tokens(r) for r in refs_tokens]}
        res = {0: [self._join_tokens(hypo_tokens)]}
        score, _ = self.scorer.compute_score(gts, res)
        return float(score)

    def __repr__(self):
        return "cider"


class Metrics:
    def __init__(self) -> None:
        self.bleu1 = BLUE(ngrams=1)
        self.bleu2 = BLUE(ngrams=2)
        self.bleu3 = BLUE(ngrams=3)
        self.bleu4 = BLUE(ngrams=4)

        self.gleu = GLEU()
        self.meteor = METEOR()  # need nltk.download('omw-1.4') sometimes

        self.all = [self.bleu1, self.bleu2, self.bleu3,
                    self.bleu4, self.gleu, self.meteor]

        # 训练阶段只算快速指标，避免拖慢训练速度
        self.train_metrics = [self.bleu4]

        # add CIDEr if available
        if _CIDER_OK:
            self.cider = CIDEr()
            self.all.append(self.cider)
        else:
            self.cider = None

    def calculate(self,
                  refs: List[List[List[str]]],
                  hypos: List[List[str]],
                  train: bool = False) -> Dict[str, float]:
        if train:
            score_fns = self.train_metrics
        else:
            score_fns = self.all

        score: Dict[str, float] = {}
        for fn in score_fns:
            score[repr(fn)] = mean(list(map(fn, refs, hypos)))

        return score
