# position_cooccurrence_graph_prediction.py
from typing import List, Tuple, Dict, DefaultDict
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_history(lottery_type_instance: lg_lottery_type, is_main: bool, limit: int = 300) -> List[List[int]]:
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

    # Build co-occurrence counts and simple position weights (rank in sorted draw)
    coocc: DefaultDict[int, Counter] = defaultdict(Counter)
    freq = Counter()
    pos_sum = Counter()
    pos_cnt = Counter()

    for draw in history:
        s = sorted([n for n in draw if min_num <= n <= max_num])
        for i, a in enumerate(s):
            freq[a] += 1
            pos_sum[a] += i
            pos_cnt[a] += 1
            for b in s:
                if a != b:
                    coocc[a][b] += 1

    # Compute average position (lower is left/low); prefer stable central positions
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        f = float(freq.get(n, 0))
        if pos_cnt.get(n, 0) > 0:
            avg_pos = pos_sum[n] / pos_cnt[n]
        else:
            avg_pos = 0.0
        # Centrality via total co-occ counts
        centrality = sum(coocc[n].values())
        # Score: freq weight + coocc weight + center preference (small penalty to extremes)
        scores[n] = 1.0 * f + 0.5 * centrality - 0.05 * abs(avg_pos)

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
