from typing import List, Tuple, Dict
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 350) -> List[List[int]]:
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


def _gap_stats(history: List[List[int]], min_num: int, max_num: int):
    # compute intervals between occurrences per number
    last_idx: Dict[int, int] = {n: None for n in range(min_num, max_num + 1)}  # type: ignore
    sums: Dict[int, float] = {n: 0.0 for n in range(min_num, max_num + 1)}
    counts: Dict[int, int] = {n: 0 for n in range(min_num, max_num + 1)}

    for idx, draw in enumerate(history):  # history newest first
        present = set(x for x in draw if min_num <= x <= max_num)
        for n in range(min_num, max_num + 1):
            if n in present:
                if last_idx[n] is not None:
                    gap = idx - last_idx[n]
                    if gap > 0:
                        sums[n] += 1.0 / gap
                        counts[n] += 1
                last_idx[n] = idx

    # harmonic mean per number; if no intervals, use global default
    hm: Dict[int, float] = {}
    default = 0.0
    # global fallback harmonic mean based on average counts
    total_s = sum(sums.values())
    total_c = sum(counts.values())
    if total_c > 0 and total_s > 0:
        default = total_c / total_s
    else:
        default = max(1.0, len(history) / 20.0)

    for n in range(min_num, max_num + 1):
        if counts[n] > 0 and sums[n] > 0:
            hm[n] = counts[n] / sums[n]
        else:
            hm[n] = default

    # current age since last occurrence
    ages: Dict[int, int] = {}
    for n in range(min_num, max_num + 1):
        ages[n] = len(history) if last_idx[n] is None else last_idx[n]

    return hm, ages


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if not count or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))
    hm, ages = _gap_stats(history, min_num, max_num)

    # Also compute long-term frequency for tie-break
    freq = {n: 0 for n in range(min_num, max_num + 1)}
    for d in history:
        for x in d:
            if min_num <= x <= max_num:
                freq[x] += 1

    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        target = hm[n]
        age = ages[n]
        # prefer age close to harmonic mean interval, then frequency, then smaller number
        scores[n] = -abs(float(age) - float(target)) + 0.001 * float(freq.get(n, 0))

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
