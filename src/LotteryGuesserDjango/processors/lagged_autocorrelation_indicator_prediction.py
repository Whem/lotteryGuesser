from typing import List, Tuple, Dict
from collections import defaultdict
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


def _acf_score(series: List[int], lags: List[int]) -> float:
    T = len(series)
    score = 0.0
    for L in lags:
        if L >= T:
            continue
        num = 0
        for t in range(L, T):
            num += series[t] * series[t - L]
        denom = max(1, T - L)
        w = 1.0 / (1.0 + L)
        score += w * (num / denom)
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
    lags = [1, 2, 3, 5, 8]
    scores: Dict[int, float] = {n: _acf_score(presence[n], lags) for n in range(min_num, max_num + 1)}
    base = [n for n, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic lagged autocorrelation indicator over 0/1 presence series for each number.
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
