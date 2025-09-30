# conditional_mutual_information_prediction.py
from typing import List, Tuple, Dict
from collections import Counter
import math
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


def _binary_presence(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, List[int]]:
    pres: Dict[int, List[int]] = {n: [] for n in range(min_num, max_num + 1)}
    for draw in history:
        s = set(n for n in draw if min_num <= n <= max_num)
        for n in range(min_num, max_num + 1):
            pres[n].append(1 if n in s else 0)
    return pres


def _mutual_information(xs: List[int], ys: List[int]) -> float:
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    c11 = c10 = c01 = c00 = 0
    for a, b in zip(xs, ys):
        if a == 1 and b == 1:
            c11 += 1
        elif a == 1 and b == 0:
            c10 += 1
        elif a == 0 and b == 1:
            c01 += 1
        else:
            c00 += 1
    eps = 1e-12
    p11 = c11 / n
    p10 = c10 / n
    p01 = c01 / n
    p00 = c00 / n
    px1 = p11 + p10
    px0 = p01 + p00
    py1 = p11 + p01
    py0 = p10 + p00
    mi = 0.0
    for pxy, px, py in [
        (p11, px1, py1),
        (p10, px1, py0),
        (p01, px0, py1),
        (p00, px0, py0),
    ]:
        if pxy > 0:
            mi += pxy * math.log(pxy / max(px * py, eps) + eps)
    return float(mi)


def _predict_block(min_num: int, max_num: int, count: int, history: List[List[int]]) -> List[int]:
    if count <= 0:
        return []
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    # Anchors: top by recent frequency
    W = min(40, len(history))
    recent = history[:W]
    freq = Counter(n for d in recent for n in d if min_num <= n <= max_num)
    anchors = [n for n, _ in freq.most_common(5)]

    pres = _binary_presence(history, min_num, max_num)

    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        s = 0.0
        xs = pres[n]
        for a in anchors:
            if a == n:
                continue
            s += _mutual_information(xs, pres[a])
        scores[n] = s

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
