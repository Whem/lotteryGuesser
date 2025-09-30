# balanced_sum_span_optimizer.py
from typing import List, Tuple
from collections import Counter
from statistics import median
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_history(lottery_type_instance: lg_lottery_type, is_main: bool, limit: int = 250) -> List[List[int]]:
    field = 'lottery_type_number' if is_main else 'additional_numbers'
    qs = (
        lg_lottery_winner_number.objects
        .filter(lottery_type=lottery_type_instance)
        .order_by('-id')
        .values_list(field, flat=True)[:limit]
    )
    out: List[List[int]] = []
    for row in qs:
        if isinstance(row, list) and row:
            out.append([int(x) for x in row if isinstance(x, int)])
    return out


def _det_fill(min_num: int, max_num: int, count: int, base: List[int]) -> List[int]:
    seen = set()
    res: List[int] = []
    for n in base:
        if min_num <= n <= max_num and n not in seen:
            seen.add(n)
            res.append(n)
            if len(res) == count:
                return sorted(res)
    for n in range(min_num, max_num + 1):
        if n not in seen:
            res.append(n)
            if len(res) == count:
                return sorted(res)
    return sorted(res)[:count]


def _target_stats(history: List[List[int]]) -> Tuple[int, int]:
    sums = [sum(d) for d in history if d]
    spans = [(max(d) - min(d)) for d in history if len(d) >= 2]
    tgt_sum = int(median(sums)) if sums else 0
    tgt_span = int(median(spans)) if spans else 0
    return tgt_sum, tgt_span


def _predict_block(min_num: int, max_num: int, count: int, history: List[List[int]]) -> List[int]:
    if count <= 0:
        return []
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    tgt_sum, tgt_span = _target_stats(history)
    tgt_mean = tgt_sum / max(1, count)

    selected: List[int] = []
    used = set()

    while len(selected) < count:
        cur_sum = sum(selected) if selected else 0
        cur_min = min(selected) if selected else None
        cur_max = max(selected) if selected else None

        best = None
        best_score = None
        for n in range(min_num, max_num + 1):
            if n in used:
                continue
            new_sum = cur_sum + n
            if cur_min is None:
                new_span = 0
            else:
                new_min = min(cur_min, n)
                new_max = max(cur_max, n)
                new_span = new_max - new_min
            # score: closeness to target mean and span
            score = abs(n - tgt_mean) + 0.1 * abs(new_span - tgt_span)
            key = (score, n)
            if best_score is None or key < best_score:
                best_score = key
                best = n
        if best is None:
            break
        selected.append(best)
        used.add(best)

    return _det_fill(min_num, max_num, count, selected)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    main_hist = _get_history(lottery_type_instance, True)
    main = _predict_block(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers),
        main_hist,
    )

    additional: List[int] = []
    if getattr(lottery_type_instance, 'has_additional_numbers', False):
        add_hist = _get_history(lottery_type_instance, False)
        additional = _predict_block(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count),
            add_hist,
        )
    return main, additional
