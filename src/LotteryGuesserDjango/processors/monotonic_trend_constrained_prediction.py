from typing import List, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 240) -> List[List[int]]:
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


def _presence_count(history: List[List[int]], min_num: int, max_num: int, window: int) -> Dict[int, int]:
    sub = history[:window]
    counts: Dict[int, int] = {n: 0 for n in range(min_num, max_num + 1)}
    for draw in sub:
        for n in draw:
            if min_num <= n <= max_num:
                counts[n] += 1
    return counts


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
    T = len(history)
    windows = [w for w in [20, 50, 100, 150] if w <= T]
    if not windows:
        windows = [T]
    density: Dict[int, List[float]] = {n: [] for n in range(min_num, max_num + 1)}
    for w in windows:
        counts = _presence_count(history, min_num, max_num, w)
        for n in range(min_num, max_num + 1):
            density[n].append(counts[n] / float(w))
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        d = density[n]
        mono = sum(1 for i in range(len(d) - 1) if d[i] <= d[i + 1])
        recent = d[0] if d else 0.0
        scores[n] = mono + 0.1 * recent
    base = [n for n, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic monotonic-trend constrained selection across multi-window densities.
    Prefers numbers whose presence density is non-decreasing across larger windows.
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
