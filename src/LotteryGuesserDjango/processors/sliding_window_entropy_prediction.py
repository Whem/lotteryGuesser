from typing import List, Tuple, Dict
from math import log2
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 300) -> List[List[int]]:
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


def _build_presence(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, List[int]]:
    presence: Dict[int, List[int]] = {n: [0] * len(history) for n in range(min_num, max_num + 1)}
    for t, draw in enumerate(history):
        s = set(n for n in draw if min_num <= n <= max_num)
        for n in s:
            presence[n][t] = 1
    return presence


def _entropy_of_prob(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


def _score_entropy(series: List[int], window_sizes: List[int]) -> float:
    T = len(series)
    score = 0.0
    for i, w in enumerate(window_sizes):
        if w > T:
            continue
        wnd = series[:w]
        p = sum(wnd) / float(w)
        # emphasize recency by larger weight for smaller windows
        weight = 1.0 / (1.0 + i)
        score += weight * _entropy_of_prob(p)
    return float(score)


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


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if count is None or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return _deterministic_fill(min_num, max_num, count, [])
    presence = _build_presence(history, min_num, max_num)
    # Window sizes from most recent draws (we use the slice [:w], history is already newest-first)
    window_sizes = [20, 50, 100, 150]
    scores: Dict[int, float] = {n: _score_entropy(presence[n], [w for w in window_sizes if w <= len(history)])
                                for n in range(min_num, max_num + 1)}
    # Prefer smaller entropy (more predictable), tie-break by number
    base = [n for n, _ in sorted(scores.items(), key=lambda x: (x[1], x[0]))[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic sliding-window entropy selector: pick numbers with lowest
    presence entropy across multiple recent windows. Returns (main, additional).
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
