from math import log2
from torch import Tensor, sort, cat


def compute_gain(y_value: float, gain_scheme: str) -> float:
    return y_value if gain_scheme == 'const' else 2 ** y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    ys_pred, indices = sort(ys_pred, descending=True)
    ys_true = ys_true[indices]

    sum_dcg = 0
    for i, y_true in enumerate(ys_true):
        gain = compute_gain(y_true, gain_scheme)
        sum_dcg += gain / log2(i + 2)
    return sum_dcg


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    case_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    return case_dcg / ideal_dcg


def compute_ideal_dcg(ys_true: Tensor, ndcg_scheme: str = 'const') -> float:
    ideal_dcg = dcg(ys_true, ys_true, ndcg_scheme)
    return ideal_dcg