from typing import List, Tuple, Dict
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 400) -> List[List[int]]:
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


def _residue_distribution(history: List[List[int]], min_num: int, max_num: int, m: int) -> Dict[int, float]:
    counts = {r: 0 for r in range(m)}
    total = 0
    for draw in history:
        for n in draw:
            if min_num <= n <= max_num:
                counts[n % m] += 1
                total += 1
    if total == 0:
        return {r: 1.0 / m for r in range(m)}
    return {r: counts[r] / float(total) for r in range(m)}


def _frequencies(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, int]:
    freq = {n: 0 for n in range(min_num, max_num + 1)}
    for draw in history:
        for n in draw:
            if min_num <= n <= max_num:
                freq[n] += 1
    return freq


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int, modulus: int = 10) -> List[int]:
    if not count or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    freq = _frequencies(history, min_num, max_num)
    m = max(3, min(12, modulus))
    dist = _residue_distribution(history, min_num, max_num, m)

    # desired counts per residue class
    desired = {r: int(round(dist[r] * count)) for r in range(m)}
    # adjust to ensure sum equals count deterministically
    total_assigned = sum(desired.values())
    if total_assigned != count:
        # compute fractional parts for fair rounding
        remainders = sorted(
            [
                (r, (dist[r] * count) - math.floor(dist[r] * count))
                for r in range(m)
            ],
            key=lambda x: (-x[1], x[0])
        )
        if total_assigned < count:
            for r, _ in remainders:
                if total_assigned == count:
                    break
                desired[r] += 1
                total_assigned += 1
        else:
            for r, _ in reversed(remainders):
                if total_assigned == count:
                    break
                if desired[r] > 0:
                    desired[r] -= 1
                    total_assigned -= 1

    # build candidate list from each residue by historical freq then number
    by_residue: Dict[int, List[int]] = {r: [] for r in range(m)}
    for n in range(min_num, max_num + 1):
        by_residue[n % m].append(n)
    base: List[int] = []
    for r in range(m):
        picks = sorted(by_residue[r], key=lambda n: (-freq.get(n, 0), n))[: desired[r]]
        base.extend(picks)
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    main = _predict_block(
        lottery_type_instance,
        'lottery_type_number',
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers),
        modulus=10,
    )
    additional: List[int] = []
    if getattr(lottery_type_instance, 'has_additional_numbers', False):
        additional = _predict_block(
            lottery_type_instance,
            'additional_numbers',
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count),
            modulus=10,
        )
    return main, additional
