from typing import List, Tuple, Dict
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


def _counts_in_window(history: List[List[int]], min_num: int, max_num: int, start: int, length: int) -> Dict[int, int]:
    end = min(len(history), start + length)
    counts: Dict[int, int] = {n: 0 for n in range(min_num, max_num + 1)}
    for draw in history[start:end]:
        for n in draw:
            if min_num <= n <= max_num:
                counts[n] += 1
    return counts


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if count is None or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return _deterministic_fill(min_num, max_num, count, [])
    # Newest-first. Compare adjacent non-overlapping windows near the front.
    windows = [20, 40, 80]
    scores: Dict[int, float] = {n: 0.0 for n in range(min_num, max_num + 1)}
    for i, w in enumerate(windows):
        if 2 * w > len(history):
            continue
        recent = _counts_in_window(history, min_num, max_num, 0, w)
        prev = _counts_in_window(history, min_num, max_num, w, w)
        weight = 1.0 / (1.0 + i)
        for n in range(min_num, max_num + 1):
            scores[n] += weight * (recent[n] - prev[n])
    base = [n for n, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic difference-profile projection: select numbers with positive
    momentum across multiple paired recent-vs-previous windows.
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
