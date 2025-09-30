# seasonal_week_booster_prediction.py
from typing import List, Tuple, Dict
from collections import Counter
import datetime
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_week_history(lottery_type_instance: lg_lottery_type, is_main: bool, week: int, limit_years: int = 10) -> List[List[int]]:
    field = 'lottery_type_number' if is_main else 'additional_numbers'
    qs = (
        lg_lottery_winner_number.objects
        .filter(lottery_type=lottery_type_instance, lottery_type_number_week=week)
        .order_by('-id')
        .values_list(field, flat=True)
    )
    out: List[List[int]] = []
    for row in qs:
        if isinstance(row, list) and row:
            out.append([int(x) for x in row if isinstance(x, int)])
    return out


def _get_all_history(lottery_type_instance: lg_lottery_type, is_main: bool, limit: int = 300) -> List[List[int]]:
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


def _predict_block(lottery_type_instance: lg_lottery_type, min_num: int, max_num: int, count: int, is_main: bool) -> List[int]:
    if count <= 0:
        return []
    today = datetime.date.today()
    _, week, _ = today.isocalendar()

    week_hist = _get_week_history(lottery_type_instance, is_main, week)
    all_hist = _get_all_history(lottery_type_instance, is_main)

    week_freq = Counter(n for d in week_hist for n in d if min_num <= n <= max_num)
    all_freq = Counter(n for d in all_hist for n in d if min_num <= n <= max_num)

    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        wf = float(week_freq.get(n, 0))
        af = float(all_freq.get(n, 0))
        scores[n] = 1.0 * wf + 0.25 * af

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [n for n, _ in ranked[:count]]
    return _det_fill(min_num, max_num, count, base)


essential_attrs = (
    'min_number', 'max_number', 'pieces_of_draw_numbers',
    'has_additional_numbers', 'additional_min_number', 'additional_max_number', 'additional_numbers_count'
)


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
