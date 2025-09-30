from typing import List, Tuple, Dict
from collections import defaultdict, Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 200) -> List[List[int]]:
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
    # Deduplicate preserving order
    seen = set()
    filtered = []
    for n in base:
        if min_num <= n <= max_num and n not in seen:
            seen.add(n)
            filtered.append(n)
    # Fill ascending
    if len(filtered) < needed:
        remain = [n for n in range(min_num, max_num + 1) if n not in seen]
        filtered.extend(remain[: needed - len(filtered)])
    return sorted(filtered)[:needed]


def _score_numbers_by_markov(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, float]:
    freq = Counter()
    out_strength = defaultdict(int)
    for draw in history:
        uniq_sorted = sorted(set(x for x in draw if min_num <= x <= max_num))
        for x in uniq_sorted:
            freq[x] += 1
        for a, b in zip(uniq_sorted, uniq_sorted[1:]):
            out_strength[a] += 1
            out_strength[b] += 1  # symmetric within-draw adjacency
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        scores[n] = float(freq.get(n, 0)) + 0.75 * float(out_strength.get(n, 0))
    return scores


def _select_top(scores: Dict[int, float], required: int) -> List[int]:
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [n for n, _ in ordered[:required]]


def _predict_block(lottery_type_instance: lg_lottery_type,
                   number_field: str,
                   min_num: int, max_num: int,
                   count: int) -> List[int]:
    if count is None or count <= 0 or min_num is None or max_num is None:
        return []
    history = _get_recent_draws(lottery_type_instance, number_field)
    if not history:
        return _deterministic_fill(min_num, max_num, count, [])
    scores = _score_numbers_by_markov(history, min_num, max_num)
    base = _select_top(scores, count)
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic Markov-like transition scoring over within-draw adjacencies.
    Returns (main_numbers, additional_numbers), both sorted and unique.
    """
    try:
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
    except Exception:
        # Fully deterministic fallback: ascending fill
        main = _deterministic_fill(
            int(lottery_type_instance.min_number),
            int(lottery_type_instance.max_number),
            int(lottery_type_instance.pieces_of_draw_numbers),
            [],
        )
        additional: List[int] = []
        if getattr(lottery_type_instance, 'has_additional_numbers', False):
            additional = _deterministic_fill(
                int(lottery_type_instance.additional_min_number),
                int(lottery_type_instance.additional_max_number),
                int(lottery_type_instance.additional_numbers_count),
                [],
            )
        return main, additional
