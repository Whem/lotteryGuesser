# diversity_set_cover_prediction.py
from typing import List, Tuple, Dict, Set
from collections import defaultdict
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


def _bands(min_num: int, max_num: int, bins: int, n: int) -> int:
    span = max(1, max_num - min_num + 1)
    idx = int((n - min_num) * bins / span)
    if idx >= bins:
        idx = bins - 1
    if idx < 0:
        idx = 0
    return idx


def _features(min_num: int, max_num: int, n: int) -> Set[str]:
    feats = set()
    feats.add(f"par:{n % 2}")
    feats.add(f"m3:{n % 3}")
    feats.add(f"m5:{n % 5}")
    feats.add(f"b5:{_bands(min_num, max_num, 5, n)}")
    return feats


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

    last_draw = set(history[0]) if history else set()

    # Coverage counters per feature category
    cov: Dict[str, int] = defaultdict(int)

    selected: List[int] = []
    used = set()

    while len(selected) < count:
        best = None
        best_key = None
        for n in range(min_num, max_num + 1):
            if n in used:
                continue
            feats = _features(min_num, max_num, n)
            # Score prefers numbers that cover features with lowest current coverage
            score = 0.0
            for f in feats:
                score += 1.0 / (1.0 + cov[f])
            # Lightly penalize immediate repetition from the very last draw
            penalty = 0.1 if n in last_draw else 0.0
            key = (-(score - penalty), n)  # minimize tuple => maximize score, tie by smaller n
            if best_key is None or key < best_key:
                best_key = key
                best = n
        if best is None:
            break
        selected.append(best)
        used.add(best)
        for f in _features(min_num, max_num, best):
            cov[f] += 1

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
