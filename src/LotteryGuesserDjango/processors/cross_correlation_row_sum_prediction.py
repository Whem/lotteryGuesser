from typing import List, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, is_main: bool, limit: int = 250) -> List[List[int]]:
    field = 'lottery_type_number' if is_main else 'additional_numbers'
    qs = (
        lg_lottery_winner_number.objects
        .filter(lottery_type=lottery_type_instance)
        .order_by('-id')
        .values_list(field, flat=True)[:limit]
    )
    out: List[List[int]] = []
    for row in qs:
        if isinstance(row, list):
            out.append([int(x) for x in row if isinstance(x, int)])
    return out


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


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return num / ((dx ** 0.5) * (dy ** 0.5))


def _predict_block(lottery_type_instance: lg_lottery_type,
                   min_num: int, max_num: int, count: int,
                   is_main: bool) -> List[int]:
    if not count or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, is_main)
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    sums = [float(sum(draw)) for draw in history]

    # presence vectors newest first
    presence: Dict[int, List[float]] = {n: [] for n in range(min_num, max_num + 1)}
    for draw in history:
        s = set(draw)
        for n in range(min_num, max_num + 1):
            presence[n].append(1.0 if n in s else 0.0)

    # Global frequency for tie-break
    freq: Dict[int, int] = {n: 0 for n in range(min_num, max_num + 1)}
    for d in history:
        for x in d:
            if min_num <= x <= max_num:
                freq[x] += 1

    # Correlation magnitude with draw sums
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        scores[n] = abs(_pearson_corr(presence[n], sums)) + 0.0005 * float(freq.get(n, 0))

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [n for n, _ in ranked[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    main = _predict_block(
        lottery_type_instance,
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers),
        True,
    )
    additional: List[int] = []
    if getattr(lottery_type_instance, 'has_additional_numbers', False):
        additional = _predict_block(
            lottery_type_instance,
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count),
            False,
        )
    return main, additional
