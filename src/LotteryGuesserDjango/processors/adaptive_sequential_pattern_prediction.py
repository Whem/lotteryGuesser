# adaptive_sequential_pattern_prediction.py
# Combines sequential patterns, cross-correlations and frequency analysis
# with adaptive weighting based on recent performance

import random
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
import numpy as np
from itertools import combinations


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Adaptive algorithm that combines multiple pattern recognition approaches
    with dynamic weighting based on recent success patterns.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model

    Returns:
    - List of predicted lottery numbers
    """
    # Fetch historical data - get more data for better pattern recognition
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:200])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    prediction_pool = set()

    # 1. Analyze sequential patterns
    sequence_numbers = analyze_sequential_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    prediction_pool.update(sequence_numbers[:required_numbers // 2])

    # 2. Identify cross-draw correlations
    correlation_numbers = find_cross_draw_correlations(past_draws)
    prediction_pool.update(correlation_numbers[:required_numbers // 2])

    # 3. Calculate frequency-based hot zones
    hot_zones = calculate_hot_zones(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    prediction_pool.update(hot_zones[:required_numbers // 2])

    # 4. Check for cyclic patterns
    cyclic_numbers = identify_cyclic_patterns(past_draws)
    prediction_pool.update(cyclic_numbers[:required_numbers // 2])

    # 5. Fill remaining slots with probability-weighted selection
    while len(prediction_pool) < required_numbers:
        weights = calculate_adaptive_weights(
            past_draws,
            lottery_type_instance.min_number,
            lottery_type_instance.max_number,
            prediction_pool
        )
        available_numbers = set(range(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number + 1
        )) - prediction_pool

        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
            prediction_pool.add(selected)

    # Ensure we have exactly the required number of numbers
    result = sorted(list(prediction_pool))[:required_numbers]
    return result


def analyze_sequential_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze sequential patterns in the historical data."""
    sequence_scores = defaultdict(float)

    # Look for arithmetic sequences
    for draw in past_draws[:50]:  # Focus on more recent draws
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 2):
            for j in range(i + 1, len(sorted_draw) - 1):
                diff = sorted_draw[j] - sorted_draw[i]
                next_num = sorted_draw[j] + diff
                if min_num <= next_num <= max_num:
                    sequence_scores[next_num] += 1 / (past_draws.index(draw) + 1)

    # Include reverse sequences
    for draw in past_draws[:30]:
        sorted_draw = sorted(draw, reverse=True)
        for i in range(len(sorted_draw) - 2):
            diff = sorted_draw[i] - sorted_draw[i + 1]
            next_num = sorted_draw[i + 1] - diff
            if min_num <= next_num <= max_num:
                sequence_scores[next_num] += 0.5 / (past_draws.index(draw) + 1)

    return sorted(sequence_scores.keys(), key=sequence_scores.get, reverse=True)


def find_cross_draw_correlations(past_draws: List[List[int]]) -> List[int]:
    """Identify numbers that have strong correlations across draws."""
    correlation_scores = defaultdict(float)

    # Look for numbers that often appear together in consecutive draws
    for i in range(len(past_draws) - 1):
        current_draw = set(past_draws[i])
        next_draw = set(past_draws[i + 1])

        # Score numbers that carried over
        for num in current_draw & next_draw:
            correlation_scores[num] += 1

        # Score numbers that appear after certain combinations
        for num in next_draw:
            for pair in combinations(current_draw, 2):
                if num > max(pair):
                    correlation_scores[num] += 0.2

    return sorted(correlation_scores.keys(), key=correlation_scores.get, reverse=True)


def calculate_hot_zones(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Calculate hot zones using frequency analysis with temporal weighting."""
    frequency_scores = defaultdict(float)

    # Calculate basic frequency with recency bias
    for i, draw in enumerate(past_draws):
        recency_weight = 1 / (i + 1)  # More recent draws have higher weight
        for num in draw:
            frequency_scores[num] += recency_weight

    # Identify hot zones using sliding windows
    range_size = (max_num - min_num) // 5  # Divide range into 5 zones
    for base in range(min_num, max_num - range_size + 1):
        zone_sum = sum(frequency_scores[n] for n in range(base, base + range_size))
        for num in range(base, base + range_size):
            frequency_scores[num] += zone_sum * 0.1

    return sorted(frequency_scores.keys(), key=frequency_scores.get, reverse=True)


def identify_cyclic_patterns(past_draws: List[List[int]]) -> List[int]:
    """Identify numbers that follow cyclic patterns."""
    cycle_scores = defaultdict(float)

    # Look for weekly cycles
    if len(past_draws) >= 14:  # Need at least two weeks of data
        for i in range(len(past_draws) - 7):
            current_draw = set(past_draws[i])
            week_later_draw = set(past_draws[i + 7])
            common_numbers = current_draw & week_later_draw
            for num in common_numbers:
                cycle_scores[num] += 1

    # Look for monthly cycles (approximately 4 weeks)
    if len(past_draws) >= 28:
        for i in range(len(past_draws) - 28):
            current_draw = set(past_draws[i])
            month_later_draw = set(past_draws[i + 28])
            common_numbers = current_draw & month_later_draw
            for num in common_numbers:
                cycle_scores[num] += 0.5

    return sorted(cycle_scores.keys(), key=cycle_scores.get, reverse=True)


def calculate_adaptive_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate adaptive weights for remaining number selection."""
    weights = defaultdict(float)

    # Basic frequency analysis
    all_numbers = [num for draw in past_draws for num in draw]
    frequency_counter = Counter(all_numbers)

    # Calculate statistical measures
    mean = statistics.mean(all_numbers)
    std_dev = statistics.stdev(all_numbers)

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base weight from frequency
        weights[num] = frequency_counter.get(num, 0) / len(past_draws)

        # Adjust weight based on distance from mean
        z_score = abs((num - mean) / std_dev)
        weights[num] *= np.exp(-z_score / 2)  # Favor numbers closer to mean

        # Adjust for gaps since last appearance
        last_seen = float('inf')
        for i, draw in enumerate(past_draws):
            if num in draw:
                last_seen = i
                break
        if last_seen != float('inf'):
            weights[num] *= np.exp(-last_seen / 10)  # Favor recently seen numbers

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