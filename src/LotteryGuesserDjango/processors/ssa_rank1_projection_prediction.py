# ssa_rank1_projection_prediction.py
from typing import List, Tuple
import numpy as np
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_history(lottery_type_instance: lg_lottery_type, is_main: bool, limit: int = 200) -> List[List[int]]:
    field = 'lottery_type_number' if is_main else 'additional_numbers'
    qs = (
        lg_lottery_winner_number.objects
        .filter(lottery_type=lottery_type_instance)
        .order_by('-id')
        .values_list(field, flat=True)[:limit]
    )
    out: List[List[int]] = []
    for row in qs:
        if isinstance(row, list) and row:
            out.append([int(x) for x in row if isinstance(x, int)])
    return out


def _presence_matrix(history: List[List[int]], min_num: int, max_num: int) -> np.ndarray:
    if not history:
        return np.zeros((0, max_num - min_num + 1), dtype=float)
    cols = max_num - min_num + 1
    M = np.zeros((len(history), cols), dtype=float)
    for i, draw in enumerate(history):
        for n in draw:
            if min_num <= n <= max_num:
                M[i, n - min_num] = 1.0
    return M


def _det_fill(min_num: int, max_num: int, count: int, base: List[int]) -> List[int]:
    seen = set()
    res: List[int] = []
    for n in base:
        if min_num <= n <= max_num and n not in seen:
            seen.add(n)
            res.append(n)
            if len(res) == count:
                return sorted(res)
    for n in range(min_num, max_num + 1):
        if n not in seen:
            res.append(n)
            if len(res) == count:
                return sorted(res)
    return sorted(res)[:count]


def _predict_block(min_num: int, max_num: int, count: int, history: List[List[int]]) -> List[int]:
    if count <= 0:
        return []
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    M = _presence_matrix(history, min_num, max_num)
    if M.size == 0:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    # Column-centered SVD (equivalent to SSA on presence matrix)
    X = M - M.mean(axis=0, keepdims=True)
    try:
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        v1 = VT[0] if VT.size else np.zeros(X.shape[1])
    except Exception:
        v1 = X.mean(axis=0)

    scores = np.abs(v1)
    ranked_idx = np.argsort(-scores, kind='mergesort')  # stable
    base = [int(idx) + min_num for idx in ranked_idx[:count]]
    return _det_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    main_hist = _get_history(lottery_type_instance, True)
    main = _predict_block(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers),
        main_hist,
    )

    additional: List[int] = []
    if getattr(lottery_type_instance, 'has_additional_numbers', False):
        add_hist = _get_history(lottery_type_instance, False)
        additional = _predict_block(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count),
            add_hist,
        )
    return main, additional
