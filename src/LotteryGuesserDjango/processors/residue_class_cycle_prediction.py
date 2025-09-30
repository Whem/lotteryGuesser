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


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if count is None or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return _deterministic_fill(min_num, max_num, count, [])
    moduli = [3, 4, 5, 7, 9]
    # Residue class scores per modulus
    residue_scores: Dict[int, Dict[int, float]] = {m: defaultdict(float) for m in moduli}
    number_freq: Counter[int] = Counter()
    for i, draw in enumerate(history):
        weight = 1.0 / (1 + i)  # recency weight, deterministic
        for n in draw:
            if min_num <= n <= max_num:
                number_freq[n] += 1
                for m in moduli:
                    residue_scores[m][n % m] += weight
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        s = 0.0
        for m in moduli:
            s += residue_scores[m].get(n % m, 0.0)
        s += 0.001 * number_freq.get(n, 0)
        scores[n] = s
    base = [n for n, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic residue-class cycle weighting across multiple moduli.
    Returns (main_numbers, additional_numbers).
    """
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
