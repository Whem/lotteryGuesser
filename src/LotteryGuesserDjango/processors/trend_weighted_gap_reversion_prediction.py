# trend_weighted_gap_reversion_prediction.py
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
from statistics import mean
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

    # newest first ordering assumed
    last_seen: Dict[int, int] = {}
    gaps: Dict[int, List[int]] = defaultdict(list)
    freq = Counter()

    prev_pos: Dict[int, int] = {}
    for idx, draw in enumerate(history):
        s = set(n for n in draw if min_num <= n <= max_num)
        for n in s:
            freq[n] += 1
            if n in prev_pos:
                gaps[n].append(idx - prev_pos[n])
            prev_pos[n] = idx
            if n not in last_seen:
                last_seen[n] = idx

    # recency frequency in last window
    W = min(20, len(history))
    recent_freq = Counter()
    for draw in history[:W]:
        for n in draw:
            if min_num <= n <= max_num:
                recent_freq[n] += 1

    N = len(history)
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        cur_gap = N if n not in last_seen else last_seen[n]
        avg_gap = mean(gaps[n]) if gaps.get(n) else 0.0
        rfreq = float(recent_freq.get(n, 0))
        # Due score: larger current gap and average gap increase score; recent repeats reduce it
        score = float(cur_gap) + 0.5 * float(avg_gap) - 0.2 * rfreq
        scores[n] = score

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
