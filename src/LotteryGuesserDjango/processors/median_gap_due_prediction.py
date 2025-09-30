from typing import List, Tuple, Dict
from collections import defaultdict
from statistics import median
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 400) -> List[List[int]]:
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


def _compute_gap_scores(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, float]:
    T = len(history)
    # process from earliest to latest
    seq = list(reversed(history))
    last_seen: Dict[int, int] = {}
    gaps: Dict[int, List[int]] = defaultdict(list)
    for idx, draw in enumerate(seq):
        s = set(n for n in draw if min_num <= n <= max_num)
        for n in s:
            if n in last_seen:
                gaps[n].append(idx - last_seen[n])
            last_seen[n] = idx
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        cur_gap = T - 1 - last_seen[n] if n in last_seen else T
        if gaps.get(n):
            med = median(gaps[n])
        else:
            med = max(1, T // 2)
        denom = med if med and med > 0 else 1
        scores[n] = float(cur_gap) / float(denom)
    return scores


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
    scores = _compute_gap_scores(history, min_num, max_num)
    base = [n for n, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic median-gap due strategy: prioritize numbers that are overdue
    relative to their historical median inter-arrival gap.
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
