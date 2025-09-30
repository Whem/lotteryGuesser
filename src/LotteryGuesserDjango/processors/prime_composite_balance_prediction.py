from typing import List, Tuple, Dict
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def _get_recent_draws(lottery_type_instance: lg_lottery_type, field: str, limit: int = 300) -> List[List[int]]:
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


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


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


def _classify_range(min_num: int, max_num: int) -> Tuple[List[int], List[int]]:
    primes: List[int] = []
    composites: List[int] = []
    for n in range(min_num, max_num + 1):
        if _is_prime(n):
            primes.append(n)
        elif n >= 1:  # treat 1 as composite class for balancing purposes
            composites.append(n)
    return primes, composites


def _frequencies(history: List[List[int]], min_num: int, max_num: int) -> Dict[int, int]:
    freq: Dict[int, int] = {n: 0 for n in range(min_num, max_num + 1)}
    for draw in history:
        for n in draw:
            if min_num <= n <= max_num:
                freq[n] += 1
    return freq


def _historical_prime_ratio(history: List[List[int]], min_num: int, max_num: int) -> float:
    if not history:
        size = max(0, max_num - min_num + 1)
        if size == 0:
            return 0.0
        primes_in_range = sum(1 for n in range(min_num, max_num + 1) if _is_prime(n))
        return primes_in_range / float(size)
    total = 0
    prime_hits = 0
    for draw in history:
        for n in draw:
            if min_num <= n <= max_num:
                total += 1
                if _is_prime(n):
                    prime_hits += 1
    if total == 0:
        size = max(0, max_num - min_num + 1)
        if size == 0:
            return 0.0
        primes_in_range = sum(1 for n in range(min_num, max_num + 1) if _is_prime(n))
        return primes_in_range / float(size)
    return prime_hits / float(total)


def _predict_block(lottery_type_instance: lg_lottery_type, field: str,
                   min_num: int, max_num: int, count: int) -> List[int]:
    if count is None or count <= 0:
        return []
    history = _get_recent_draws(lottery_type_instance, field)
    primes, composites = _classify_range(min_num, max_num)
    if not primes and not composites:
        return []
    ratio = _historical_prime_ratio(history, min_num, max_num)
    target_primes = int(round(ratio * count))
    target_primes = max(0, min(target_primes, min(len(primes), count)))
    target_composites = count - target_primes
    target_composites = max(0, min(target_composites, len(composites)))
    # adjust if still short due to class sizes
    if target_primes + target_composites < count:
        # fill the deficit from whichever class has remaining numbers
        deficit = count - (target_primes + target_composites)
        extra_from_primes = min(deficit, max(0, len(primes) - target_primes))
        target_primes += extra_from_primes
        target_composites += deficit - extra_from_primes
    freq = _frequencies(history, min_num, max_num)
    primes_sorted = sorted(primes, key=lambda n: (-freq.get(n, 0), n))
    composites_sorted = sorted(composites, key=lambda n: (-freq.get(n, 0), n))
    base: List[int] = primes_sorted[:target_primes] + composites_sorted[:target_composites]
    return _deterministic_fill(min_num, max_num, count, base)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deterministic prime/composite balance predictor.
    Matches the historical prime ratio (or in-range prime density if no history)
    while prioritizing historically frequent numbers. Returns (main, additional).
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
