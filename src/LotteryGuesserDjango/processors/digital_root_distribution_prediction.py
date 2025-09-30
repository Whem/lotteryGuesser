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


def _digital_root(n: int) -> int:
    if n == 0:
        return 0
    r = n % 9
    return 9 if r == 0 else r


def _root_distribution(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, float]:
    counts = {r: 0 for r in range(10)}  # 0..9 roots
    total = 0
    for draw in history:
        for x in draw:
            if min_num <= x <= max_num:
                counts[_digital_root(x)] += 1
                total += 1
    if total == 0:
        # uniform over roots present in range
        present_roots = set(_digital_root(x) for x in range(min_num, max_num + 1))
        size = len(present_roots)
        if size == 0:
            return {r: 0.0 for r in range(10)}
        return {r: (1.0 / size if r in present_roots else 0.0) for r in range(10)}
    return {r: counts[r] / float(total) for r in range(10)}


def _frequencies(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, int]:
    freq = {n: 0 for n in range(min_num, max_num + 1)}
    for draw in history:
        for x in draw:
            if min_num <= x <= max_num:
                freq[x] += 1
    return freq


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if not count or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    dist = _root_distribution(history, min_num, max_num)
    freq = _frequencies(history, min_num, max_num)

    # desired per-root counts using fair rounding and deterministic tie-break by root asc
    # consider only roots present in range
    roots_in_range = sorted(set(_digital_root(x) for x in range(min_num, max_num + 1)))
    raw = {r: dist.get(r, 0.0) * count for r in roots_in_range}
    desired = {r: int(raw[r]) for r in roots_in_range}
    assigned = sum(desired.values())
    remainders = sorted(((r, raw[r] - desired[r]) for r in roots_in_range), key=lambda t: (-t[1], t[0]))
    i = 0
    while assigned < count and i < len(remainders):
        r = remainders[i][0]
        desired[r] += 1
        assigned += 1
        i += 1

    # build candidates per root by historical frequency then numeric asc
    base: List[int] = []
    for r in roots_in_range:
        candidates = [x for x in range(min_num, max_num + 1) if _digital_root(x) == r]
        picks = sorted(candidates, key=lambda x: (-freq.get(x, 0), x))[:desired[r]]
        base.extend(picks)

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
