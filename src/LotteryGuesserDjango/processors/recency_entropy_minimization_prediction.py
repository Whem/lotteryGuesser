# recency_entropy_minimization_prediction.py
from typing import List, Tuple, Dict
from collections import Counter
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


def _predict_block(min_num: int, max_num: int, count: int, history: List[List[int]]) -> List[int]:
    if count <= 0:
        return []
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    W = min(30, len(history))
    recent = history[:W]
    recent_freq = Counter(n for d in recent for n in d if min_num <= n <= max_num)
    global_freq = Counter(n for d in history for n in d if min_num <= n <= max_num)
    last_draw = set(history[0]) if history else set()

    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        r = float(recent_freq.get(n, 0))
        g = float(global_freq.get(n, 0))
        # Favor numbers that reduce uncertainty by following recent mode, penalize immediate repeats lightly
        penalty = 0.2 if n in last_draw else 0.0
        scores[n] = 1.5 * r + 0.5 * g - penalty

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [n for n, _ in ranked[:count]]
    return _det_fill(min_num, max_num, count, base)


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
