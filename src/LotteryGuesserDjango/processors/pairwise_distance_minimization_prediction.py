from typing import List, Tuple, Dict
import statistics
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
            arr = sorted(int(x) for x in row if isinstance(x, int))
            if arr:
                draws.append(arr)
    return draws


def _deterministic_fill(min_num: int, max_num: int, needed: int, base: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for n in base:
        if min_num <= n <= max_num and n not in seen:
            seen.add(n)
            out.append(n)
            if len(out) == needed:
                return sorted(out)
    for n in range(min_num, max_num + 1):
        if n not in seen:
            out.append(n)
            if len(out) == needed:
                return sorted(out)
    return sorted(out)[:needed]


def _median_avg_gap(history: List[List[int]]) -> float:
    gaps: List[float] = []
    for draw in history:
        if len(draw) > 1:
            diffs = [b - a for a, b in zip(draw[:-1], draw[1:])]
            gaps.append(sum(diffs) / (len(diffs)))
    if not gaps:
        return 1.0
    return statistics.median(gaps)


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if not count or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    d = _median_avg_gap(history)
    d = max(1.0, min(float(max_num - min_num) / max(1, count - 1), d))
    # Ideal targets equally spaced by d from min
    targets = [round(min_num + i * d) for i in range(count)]
    # Score each number by closeness to nearest target (smaller distance is better)
    # Also softly favor historically frequent numbers
    freq = {n: 0 for n in range(min_num, max_num + 1)}
    for draw in history:
        for n in draw:
            if min_num <= n <= max_num:
                freq[n] += 1
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        dist = min(abs(n - t) for t in targets) if targets else 0.0
        scores[n] = -float(dist) + 0.001 * float(freq.get(n, 0))
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [n for n, _ in ranked[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
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
