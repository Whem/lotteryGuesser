from typing import List, Tuple, Dict
import statistics
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 400) -> List[List[int]]:
    qs = (
        lg_lottery_winner_number.objects
        .filter(lottery_type=lottery_type_instance)
        .order_by('-id')
        .values_list(field, flat=True)[:limit]
    )
    out: List[List[int]] = []
    for row in qs:
        if isinstance(row, list):
            out.append([int(x) for x in row if isinstance(x, int)])
    return out


def _deterministic_fill(min_num: int, max_num: int, needed: int, base: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for n in base:
        if min_num <= n <= max_num and n not in seen:
            seen.add(n)
            out.append(n)
            if len(out) == needed:
                return sorted(out)
    for n in range(min_num, max_num + 1):
        if n not in seen:
            out.append(n)
            if len(out) == needed:
                return sorted(out)
    return sorted(out)[:needed]


def _window_presence_ratios(history: List[List[int]], min_num: int, max_num: int, windows=(30, 80, 160)) -> Dict[int, Tuple[float, float]]:
    ratios: Dict[int, List[float]] = {n: [] for n in range(min_num, max_num + 1)}
    for w in windows:
        segment = history[:w] if len(history) >= w else history
        denom = max(1, len(segment))
        counts = {n: 0 for n in range(min_num, max_num + 1)}
        for draw in segment:
            s = set(draw)
            for n in range(min_num, max_num + 1):
                if n in s:
                    counts[n] += 1
        for n in range(min_num, max_num + 1):
            ratios[n].append(counts[n] / float(denom))
    stats: Dict[int, Tuple[float, float]] = {}
    for n, vals in ratios.items():
        m = sum(vals) / len(vals) if vals else 0.0
        v = statistics.pvariance(vals) if len(vals) >= 2 else 0.0
        stats[n] = (m, v)
    return stats


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if not count or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    if not history:
        return list(range(min_num, min(min_num + count, max_num + 1)))

    stats = _window_presence_ratios(history, min_num, max_num)
    # Prefer low variance (stable appearance) and higher mean presence
    scores: Dict[int, float] = {}
    for n in range(min_num, max_num + 1):
        mean_r, var_r = stats[n]
        scores[n] = -var_r + 0.5 * mean_r

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    base = [n for n, _ in ranked[:count]]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
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
