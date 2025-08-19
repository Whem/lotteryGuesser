# adaptive_hybrid_prediction.py
import numpy as np
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from collections import defaultdict


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Adaptive hybrid prediction algorithm for both simple and combined lottery types.
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
    """Generate a set of numbers using hybrid prediction methods."""
    # Get past draws
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])

    # Extract appropriate number sets from past draws
    if is_main:
        past_numbers = [draw.lottery_type_number for draw in past_draws
                        if isinstance(draw.lottery_type_number, list)]
    else:
        past_numbers = [draw.additional_numbers for draw in past_draws
                        if hasattr(draw, 'additional_numbers') and
                        isinstance(draw.additional_numbers, list)]

    if len(past_numbers) < 20:
        return deterministic_fallback(min_num, max_num, required_numbers)

    # Define prediction methods
    methods = [
        # Method 1: Top frequency selection (deterministic)
        lambda: select_top_frequency_numbers(
            past_numbers, min_num, max_num, required_numbers
        ),

        # Method 2: Use most recent draw (deterministic fallback if unusable)
        lambda: (sorted(past_numbers[0])[:required_numbers]
                 if past_numbers and len(past_numbers[0]) >= required_numbers
                 else deterministic_fallback(min_num, max_num, required_numbers)),

        # Method 3: Historical frequency-based selection (same as Method 1 to keep deterministic)
        lambda: select_top_frequency_numbers(
            past_numbers, min_num, max_num, required_numbers
        ),

        # Method 4: Mean-based selection
        lambda: generate_mean_based_numbers(
            past_numbers,
            min_num,
            max_num,
            required_numbers
        ),

        # Method 5: Pattern-based selection
        lambda: generate_pattern_based_numbers(
            past_numbers,
            min_num,
            max_num,
            required_numbers
        )
    ]

    # Initialize weights
    weights = [1.0 / len(methods)] * len(methods)

    # Generate predictions from each method
    predictions = []
    for method in methods:
        try:
            pred = method()
            if len(pred) == required_numbers:
                predictions.append(pred)
            else:
                # Deterministic fallback if method fails
                predictions.append(deterministic_fallback(min_num, max_num, required_numbers))
        except Exception as e:
            print(f"Method error: {e}")
            # Deterministic fallback
            predictions.append(deterministic_fallback(min_num, max_num, required_numbers))

    # Evaluate recent performance
    if past_numbers:
        recent_draw = past_numbers[0]
        performances = [len(set(pred) & set(recent_draw)) for pred in predictions]

        # Update weights based on performance
        total_performance = sum(performances)
        if total_performance > 0:
            weights = [p / total_performance for p in performances]
        else:
            weights = [1.0 / len(methods)] * len(methods)

    # Deterministic weighted aggregation: sum weights per number and pick top-K
    score = defaultdict(float)
    for idx, pred in enumerate(predictions):
        w = float(weights[idx]) if idx < len(weights) else 0.0
        for n in pred:
            if min_num <= n <= max_num:
                score[int(n)] += w

    ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
    final = [n for n, _ in ranked][:required_numbers]

    # Fill deterministically with smallest remaining numbers
    if len(final) < required_numbers:
        existing = set(final)
        for candidate in range(min_num, max_num + 1):
            if candidate not in existing:
                final.append(candidate)
                existing.add(candidate)
                if len(final) >= required_numbers:
                    break

    return sorted(final[:required_numbers])


def generate_mean_based_numbers(
        past_numbers: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on historical means."""
    if not past_numbers:
        return deterministic_fallback(min_num, max_num, required_numbers)

    # Calculate mean for each position
    position_means = []
    for pos in range(required_numbers):
        position_numbers = [draw[pos] for draw in past_numbers if len(draw) > pos]
        if position_numbers:
            position_means.append(int(round(np.mean(position_numbers))))
        else:
            position_means.append(min_num + pos)

    # Adjust means to ensure unique numbers within bounds
    result: List[int] = []
    used = set()
    # Deterministic offset sequence: 0, +1, -1, +2, -2, ...
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

    for mean in position_means:
        for off in offsets():
            candidate = mean + off
            if min_num <= candidate <= max_num and candidate not in used:
                result.append(int(candidate))
                used.add(int(candidate))
                break

    return sorted(result[:required_numbers])


def generate_pattern_based_numbers(
        past_numbers: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on common patterns in past draws."""
    if not past_numbers or len(past_numbers) < 2:
        return deterministic_fallback(min_num, max_num, required_numbers)

    # Calculate common differences between consecutive numbers
    differences = []
    for draw in past_numbers:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            differences.append(sorted_draw[i + 1] - sorted_draw[i])

    # Use median difference to generate new numbers
    median_diff = int(np.median(differences)) if differences else 1

    # Start with deterministic first number: smallest from most recent draw if available, else min_num
    result: List[int] = []
    used = set()
    first_draw = sorted(past_numbers[0]) if past_numbers and past_numbers[0] else []
    start_num = first_draw[0] if first_draw else min_num
    start_num = max(min_num, min(max_num, start_num))
    result.append(int(start_num))
    used.add(int(start_num))

    # Generate subsequent numbers using the pattern deterministically
    current = start_num
    while len(result) < required_numbers:
        next_num = current + median_diff
        if next_num > max_num:
            # Restart from smallest available within range
            for candidate in range(min_num, max_num + 1):
                if candidate not in used:
                    next_num = candidate
                    break
        if next_num < min_num:
            next_num = min_num
        if next_num not in used:
            result.append(int(next_num))
            used.add(int(next_num))
        current = next_num

    return sorted(result[:required_numbers])


def select_top_frequency_numbers(past_numbers: List[List[int]], min_num: int, max_num: int, k: int) -> List[int]:
    from collections import Counter
    freq = Counter()
    for draw in past_numbers:
        for n in draw:
            if min_num <= int(n) <= max_num:
                freq[int(n)] += 1
    if not freq:
        return deterministic_fallback(min_num, max_num, k)
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [n for n, _ in ranked][:k]


def deterministic_fallback(min_num: int, max_num: int, count: int) -> List[int]:
    """Return the smallest 'count' numbers within [min_num, max_num]."""
    span = max_num - min_num + 1
    if count <= 0 or span <= 0:
        return []
    if count >= span:
        return [int(x) for x in range(min_num, max_num + 1)][:count]
    return [int(x) for x in range(min_num, min_num + count)]