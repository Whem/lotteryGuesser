# ensemble_prediction.py
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import numpy as np


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Ensemble predictor for combined lottery types.
    Returns (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using ensemble prediction."""
    # Collect predictions from all models
    predictions = []

    # Generate simple ensemble predictions using deterministic methods
    historical_data = get_historical_data_simple(lottery_type_instance, is_main)

    # Define deterministic methods
    methods = [
        lambda: generate_weighted_deterministic_prediction(historical_data, min_num, max_num, required_numbers, decay=None),
        lambda: generate_weighted_deterministic_prediction(historical_data, min_num, max_num, required_numbers, decay=0.95),
        lambda: generate_weighted_deterministic_prediction(historical_data, min_num, max_num, required_numbers, decay=0.90),
        lambda: generate_mean_based_numbers(historical_data, min_num, max_num, required_numbers),
        lambda: generate_pattern_based_numbers(historical_data, min_num, max_num, required_numbers),
        lambda: deterministic_fallback(min_num, max_num, required_numbers),
    ]

    predictions = []
    for m in methods:
        try:
            pred = m()
            predictions.append(pred if len(pred) == required_numbers else deterministic_fallback(min_num, max_num, required_numbers))
        except Exception:
            predictions.append(deterministic_fallback(min_num, max_num, required_numbers))

    # Count all predicted numbers
    all_predicted_numbers = [num for pred in predictions for num in pred]
    number_counts = Counter(all_predicted_numbers)

    # Select most common numbers within valid range
    predicted_numbers = []
    for num, _ in number_counts.most_common():
        if min_num <= num <= max_num:
            predicted_numbers.append(num)
            if len(predicted_numbers) >= required_numbers:
                break

    # Fill with historical numbers if needed
    if len(predicted_numbers) < required_numbers:
        historical_numbers = get_historical_numbers(
            lottery_type_instance,
            min_num,
            max_num,
            required_numbers,
            is_main,
            predicted_numbers
        )
        predicted_numbers.extend(historical_numbers)

    # Fill deterministically if still short
    if len(predicted_numbers) < required_numbers:
        existing = set(predicted_numbers)
        for candidate in range(min_num, max_num + 1):
            if candidate not in existing:
                predicted_numbers.append(candidate)
                existing.add(candidate)
            if len(predicted_numbers) >= required_numbers:
                break

    # Final sorting and trimming
    return sorted(predicted_numbers[:required_numbers])


def get_historical_numbers(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool,
        existing_numbers: List[int]
) -> List[int]:
    """Get additional numbers from historical data."""
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id')

    all_numbers = []
    for draw in past_draws:
        numbers = draw.lottery_type_number if is_main else draw.additional_numbers
        if isinstance(numbers, list):
            all_numbers.extend(numbers)

    # Count and filter numbers
    number_counts = Counter(all_numbers)
    additional_numbers = []

    for num, _ in number_counts.most_common():
        if (min_num <= num <= max_num and
                num not in existing_numbers and
                num not in additional_numbers):
            additional_numbers.append(num)
            if len(existing_numbers) + len(additional_numbers) >= required_numbers:
                break

    return additional_numbers


def get_historical_data_simple(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get simple historical data."""
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:50]
    
    results = []
    for draw in past_draws:
        numbers = draw.lottery_type_number if is_main else getattr(draw, 'additional_numbers', [])
        if isinstance(numbers, list):
            results.append(numbers)
    
    return results


def generate_weighted_deterministic_prediction(historical_data: List[List[int]], min_num: int, max_num: int, count: int, decay: float | None = None) -> List[int]:
    """Generate deterministic weighted predictions based on historical frequency and optional temporal decay."""
    frequency = Counter()
    if decay is None:
        # Equal weight for all draws
        for draw in historical_data:
            for num in draw:
                if min_num <= num <= max_num:
                    frequency[num] += 1
    else:
        for i, draw in enumerate(historical_data):
            weight = decay ** i
            for num in draw:
                if min_num <= num <= max_num:
                    frequency[num] += weight

    ranked = sorted(frequency.items(), key=lambda x: (-x[1], x[0]))
    return [n for n, _ in ranked][:count] if ranked else deterministic_fallback(min_num, max_num, count)


def generate_mean_based_numbers(historical_data: List[List[int]], min_num: int, max_num: int, count: int) -> List[int]:
    if not historical_data:
        return deterministic_fallback(min_num, max_num, count)
    # Position-wise mean
    means = []
    for pos in range(count):
        values = [draw[pos] for draw in historical_data if len(draw) > pos and min_num <= draw[pos] <= max_num]
        if values:
            means.append(int(round(np.mean(values))))
        else:
            means.append(min_num + pos)
    # Adjust to be unique and within bounds using deterministic offsets
    selected = []
    used = set()
    def offsets():
        k = 0
        while True:
            if k == 0:
                yield 0
                k = 1
            else:
                yield k
                yield -k
                k += 1
    for m in means:
        for off in offsets():
            c = m + off
            if min_num <= c <= max_num and c not in used:
                selected.append(int(c))
                used.add(int(c))
                break
    return sorted(selected[:count])


def generate_pattern_based_numbers(historical_data: List[List[int]], min_num: int, max_num: int, count: int) -> List[int]:
    if not historical_data:
        return deterministic_fallback(min_num, max_num, count)
    diffs = []
    for draw in historical_data:
        s = sorted([n for n in draw if min_num <= n <= max_num])
        for i in range(len(s) - 1):
            diffs.append(s[i + 1] - s[i])
    step = int(np.median(diffs)) if diffs else 1
    # Start from smallest of most recent draw or min_num
    recent = sorted([n for n in historical_data[0] if min_num <= n <= max_num]) if historical_data else []
    start = recent[0] if recent else min_num
    start = max(min_num, min(max_num, start))
    res = [int(start)]
    used = {int(start)}
    cur = start
    while len(res) < count:
        nxt = cur + step
        if nxt > max_num:
            # restart from smallest available
            for cand in range(min_num, max_num + 1):
                if cand not in used:
                    nxt = cand
                    break
        if nxt < min_num:
            nxt = min_num
        if nxt not in used:
            res.append(int(nxt))
            used.add(int(nxt))
        cur = nxt
    return sorted(res[:count])


def deterministic_fallback(min_num: int, max_num: int, count: int) -> List[int]:
    span = max_num - min_num + 1
    if count <= 0 or span <= 0:
        return []
    if count >= span:
        return [int(x) for x in range(min_num, max_num + 1)][:count]
    return [int(x) for x in range(min_num, min_num + count)]