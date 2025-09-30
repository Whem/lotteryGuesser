from typing import List, Tuple, Dict
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


def _window_freq_ratio(history: List[List[int]], min_num: int, max_num: int, w: int) -> Dict[int, float]:
    seg = history[:w] if len(history) >= w else history
    denom = max(1, len(seg))
    counts = {n: 0 for n in range(min_num, max_num + 1)}
    for draw in seg:
        for x in draw:
            if min_num <= x <= max_num:
                counts[x] += 1
    return {n: counts[n] / float(denom) for n in range(min_num, max_num + 1)}


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int,
                   short_w: int = 50, long_w: int = 200) -> List[int]:
    if not count or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    short_r = _window_freq_ratio(history, min_num, max_num, short_w)
    long_r = _window_freq_ratio(history, min_num, max_num, long_w)

    # mean reversion: prefer numbers currently below their long-term ratio
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        scores[n] = (long_r.get(n, 0.0) - short_r.get(n, 0.0))

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
