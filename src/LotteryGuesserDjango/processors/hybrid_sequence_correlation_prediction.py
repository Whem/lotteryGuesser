# hybrid_sequence_correlation_prediction.py
# Hybrid algorithm combining Fibonacci patterns, pair correlations, and empirical decomposition

import random
from typing import List, Set, Tuple
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from itertools import combinations
import statistics
import numpy as np


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Hybrid predictor for combined lottery types."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

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
    """Generate numbers using hybrid analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    # Hybrid analysis components
    pair_frequencies = analyze_pair_frequencies(past_draws)
    fibonacci_candidates = generate_fibonacci_candidates(
        past_draws,
        min_num,
        max_num
    )
    empirical_patterns = analyze_empirical_patterns(past_draws)

    # Calculate allocations
    pairs_count = max(1, required_numbers // 3)
    fibonacci_count = max(1, required_numbers // 4)
    empirical_count = max(1, required_numbers // 4)

    # Generate predictions
    predicted_numbers = generate_predictions(
        pair_frequencies,
        fibonacci_candidates,
        empirical_patterns,
        past_draws,
        min_num,
        max_num,
        required_numbers,
        pairs_count,
        fibonacci_count,
        empirical_count
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def generate_predictions(
        pair_frequencies: Counter,
        fibonacci_candidates: List[int],
        empirical_patterns: List[int],
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int,
        pairs_count: int,
        fibonacci_count: int,
        empirical_count: int
) -> List[int]:
    """Generate predictions using hybrid approach."""
    predicted_numbers = set()

    # Add pair numbers
    top_pairs = sorted(pair_frequencies.items(), key=lambda x: x[1], reverse=True)
    for pair, _ in top_pairs[:pairs_count]:
        predicted_numbers.update(pair)

    # Add Fibonacci numbers
    if fibonacci_candidates:
        for num in fibonacci_candidates[:fibonacci_count]:
            predicted_numbers.add(num)

    # Add empirical numbers
    if empirical_patterns:
        for num in empirical_patterns[:empirical_count]:
            predicted_numbers.add(num)

    # Fill remaining with weighted selection
    weights = calculate_number_weights(past_draws, min_num, max_num)

    while len(predicted_numbers) < required_numbers:
        available = set(range(min_num, max_num + 1)) - predicted_numbers
        if not available:
            break

        available_weights = [weights.get(num, 1) for num in available]
        selected = random.choices(list(available), weights=available_weights, k=1)[0]
        predicted_numbers.add(selected)

    return list(predicted_numbers)[:required_numbers]


def analyze_pair_frequencies(past_draws: List[List[int]]) -> Counter:
    """Analyze frequency of number pairs in historical draws."""
    pair_counter = Counter()
    for draw in past_draws:
        for pair in combinations(sorted(draw), 2):
            pair_counter[pair] += 1
    return pair_counter


def generate_fibonacci_candidates(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Generate Fibonacci-like sequences based on historical patterns."""
    sequences = []
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 2):
            if abs((sorted_draw[i + 2] - sorted_draw[i + 1]) - (sorted_draw[i + 1] - sorted_draw[i])) <= 2:
                next_num = sorted_draw[i + 2] + (sorted_draw[i + 2] - sorted_draw[i + 1])
                if min_num <= next_num <= max_num:
                    sequences.append(next_num)

    return list(set(sequences))


def analyze_empirical_patterns(past_draws: List[List[int]]) -> List[int]:
    """Analyze empirical patterns in the data."""
    # Calculate moving averages and trends
    all_numbers = [num for draw in past_draws for num in draw]

    # Find numbers that appear in winning combinations more frequently
    number_freq = Counter(all_numbers)

    # Calculate statistical measures
    mean = statistics.mean(all_numbers)
    std_dev = statistics.stdev(all_numbers)

    # Find numbers that are within one standard deviation of the mean
    balanced_numbers = [
        num for num, freq in number_freq.items()
        if mean - std_dev <= num <= mean + std_dev
    ]

    return sorted(balanced_numbers)


def calculate_number_weights(past_draws: List[List[int]], min_num: int, max_num: int) -> dict:
    """Calculate weights for each number based on historical frequency."""
    number_freq = Counter(num for draw in past_draws for num in draw)
    total_draws = len(past_draws)

    weights = {}
    for num in range(min_num, max_num + 1):
        # Combine frequency with recency bias
        frequency_weight = number_freq.get(num, 0) / total_draws
        recency_weight = sum(1 for draw in past_draws[:10] if num in draw) / 10
        weights[num] = (frequency_weight + recency_weight) / 2

    return weights


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        numbers.add(random.randint(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number
        ))
    return sorted(list(numbers))