from typing import List, Tuple, Dict
import numpy as np
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 250) -> List[List[int]]:
    qs = (
        lg_lottery_winner_number.objects
        .filter(lottery_type=lottery_type_instance)
        .order_by('-id')
        .values_list(field, flat=True)[:limit]
    )
    draws: List[List[int]] = []
    for row in qs:
        if isinstance(row, list):
            draws.append([int(x) for x in row if isinstance(x, int)])
    return draws


def _deterministic_fill(min_num: int, max_num: int, needed: int, base: List[int]) -> List[int]:
    seen = set()
    filtered: List[int] = []
    for n in base:
        if min_num <= n <= max_num and n not in seen:
            seen.add(n)
            filtered.append(n)
    if len(filtered) < needed:
        remain = [n for n in range(min_num, max_num + 1) if n not in seen]
        filtered.extend(remain[: needed - len(filtered)])
    return sorted(filtered)[:needed]


def _build_cooccurrence_matrix(min_num: int, max_num: int, history: List[List[int]]) -> np.ndarray:
    size = max(0, max_num - min_num + 1)
    A = np.zeros((size, size), dtype=float)
    if size == 0:
        return A
    for draw in history:
        uniq = sorted(set(n for n in draw if min_num <= n <= max_num))
        for i in range(len(uniq)):
            ai = uniq[i] - min_num
            for j in range(i + 1, len(uniq)):
                aj = uniq[j] - min_num
                A[ai, aj] += 1.0
                A[aj, ai] += 1.0
    return A


def _eigenvector_centrality(A: np.ndarray, iters: int = 40) -> np.ndarray:
    n = A.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float)
    v = np.ones((n,), dtype=float)
    for _ in range(iters):
        v_next = A.dot(v)
        norm = np.linalg.norm(v_next, 1)
        if norm == 0:
            # fallback to degree centrality
            v_next = A.sum(axis=1)
            norm = np.linalg.norm(v_next, 1)
            if norm == 0:
                return v
        v = v_next / norm
    return v


def _select_top(scores: Dict[int, float], k: int) -> List[int]:
    return [n for n, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if count is None or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return _deterministic_fill(min_num, max_num, count, [])
    A = _build_cooccurrence_matrix(min_num, max_num, history)
    v = _eigenvector_centrality(A)
    scores: Dict[int, float] = {min_num + i: float(v[i]) for i in range(len(v))}
    base = _select_top(scores, count)
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic spectral co-occurrence graph predictor using power iteration
    eigenvector centrality on number co-occurrence within draws.
    Returns (main_numbers, additional_numbers).
    """
    main = _predict_block(
        lottery_type_instance,
        'lottery_type_number',
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers),
    )
    additional: List[int] = []
    if getattr(lottery_type_instance, 'has_additional_numbers', False):
        additional = _predict_block(
            lottery_type_instance,
            'additional_numbers',
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count),
        )
    return main, additional
