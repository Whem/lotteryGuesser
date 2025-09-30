# streak_breaker_prediction.py
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


def _streaks(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, int]:
    # newest first history
    streak: Dict[int, int] = {n: 0 for n in range(min_num, max_num + 1)}
    for n in range(min_num, max_num + 1):
        s = 0
        for draw in history:
            if n in draw:
                s += 1
            else:
                break
        streak[n] = s
    return streak


def _predict_block(min_num: int, max_num: int, count: int, history: List[List[int]]) -> List[int]:
    if count <= 0:
        return []
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    W = min(30, len(history))
    recent = history[:W]
    recent_freq = Counter(n for d in recent for n in d if min_num <= n <= max_num)
    global_freq = Counter(n for d in history for n in d if min_num <= n <= max_num)
    streak = _streaks(history, min_num, max_num)

    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        rf = float(recent_freq.get(n, 0))
        gf = float(global_freq.get(n, 0))
        st = float(streak.get(n, 0))
        # Anti-run: penalize long current streaks strongly
        scores[n] = 0.6 * rf + 0.1 * gf - 1.2 * st

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
