from math import log2
from torch import Tensor, sort, cat


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    ys_pred, indices = sort(ys_pred, descending=True, dim=0)
    ys_true = ys_true[indices]

    num_incorrect_pairs = 0
    len_preds = ys_true.shape[0]
    for i in range(len_preds - 1):
        for j in range(i + 1, len_preds):
            if ys_true[i] < ys_true[j]:
                num_incorrect_pairs += 1
    return num_incorrect_pairs


def compute_gain(y_value: float, gain_scheme: str) -> float:
    return y_value if gain_scheme == 'const' else 2 ** y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    ys_pred, indices = sort(ys_pred, descending=True, dim=0)
    ys_true = ys_true[indices]

    sum_dcg = 0
    for i, y_true in enumerate(ys_true, 1):
        gain = compute_gain(y_true.item(), gain_scheme)
        sum_dcg += gain / log2(i + 1)
    return sum_dcg


def ndcg(ys_true: Tensor, ys_pred: Tensor,
         gain_scheme: str = 'const') -> float:
    case_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    return case_dcg / ideal_dcg


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return -1
    ys_pred, indices = sort(ys_pred, descending=True, dim=0)
    ys_true = ys_true[indices]
    pr_at_k = ys_true[:k].sum().item() / min(ys_true.sum(), k)
    return pr_at_k


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    ys_pred, indices = sort(ys_pred, descending=True, dim=0)
    ys_true = ys_true[indices]

    ind_one = (ys_true == 1).nonzero(as_tuple=True)[0].item()
    return 1 / (ind_one + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    _, indices = sort(ys_pred, descending=True, dim=0)
    ys_true = ys_true[indices]

    p_found_ = 0
    p_look = 1
    for y_true in ys_true:
        p_found_ += p_look * y_true.item()
        p_look = p_look * (1 - y_true.item()) * (1 - p_break)
    return p_found_


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    ys_pred, indices = sort(ys_pred, descending=True, dim=0)
    ys_true = ys_true[indices]

    ap = 0
    for i in range(len(ys_true)):
        if ys_true[i].item() == 1:
            ap += ys_true[:i + 1].sum().item() / (i + 1)
    return ap / ys_true.sum().item()
