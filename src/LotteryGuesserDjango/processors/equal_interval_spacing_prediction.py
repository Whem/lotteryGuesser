from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _deterministic_fill(min_num: int, max_num: int, needed: int, base: List[int]) -> List[int]:
    seen = set()
    filtered: List[int] = []
    for n in base:
        if min_num <= n <= max_num and n not in seen:
            seen.add(n)
            filtered.append(n)
    if len(filtered) < needed:
        remain = [n for n in range(min_num, max_num + 1) if n not in seen]
        filtered.extend(remain[: needed - len(filtered)])
    return sorted(filtered)[:needed]


def _equal_spaced(min_num: int, max_num: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if count == 1:
        return [int(round((min_num + max_num) / 2))]
    step = (max_num - min_num) / float(count - 1)
    base = [int(round(min_num + i * step)) for i in range(count)]
    return _deterministic_fill(min_num, max_num, count, base)


def _predict_block(min_num: int, max_num: int, count: int) -> List[int]:
    return _equal_spaced(min_num, max_num, count)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic equal-interval spacing across the number range.
    No historical data needed; purely structural diversity, reproducible.
    """
    main = _predict_block(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers),
    )
    additional: List[int] = []
    if getattr(lottery_type_instance, 'has_additional_numbers', False):
        additional = _predict_block(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count),
        )
    return main, additional
