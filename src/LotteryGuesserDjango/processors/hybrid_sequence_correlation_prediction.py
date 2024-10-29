# hybrid_sequence_correlation_prediction.py
# Hybrid algorithm combining Fibonacci patterns, pair correlations, and empirical decomposition

import random
from typing import List, Set
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from itertools import combinations
import statistics
import numpy as np


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Hybrid algorithm combining Fibonacci patterns, pair correlations, and empirical decomposition
    to predict lottery numbers.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model

    Returns:
    - List of predicted lottery numbers based on lottery_type_instance.pieces_of_draw_numbers
    """
    required_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Get historical data
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:100]  # Last 100 draws

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]

    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    # 1. Find frequently occurring pairs
    pair_frequencies = analyze_pair_frequencies(past_draws)

    # 2. Generate Fibonacci-like sequences from historical data
    fibonacci_candidates = generate_fibonacci_candidates(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )

    # 3. Analyze empirical patterns
    empirical_patterns = analyze_empirical_patterns(past_draws)

    # Initialize predicted numbers set
    predicted_numbers = set()

    # Dynamically calculate how many numbers to take from each method based on required_numbers
    pairs_to_take = max(1, required_numbers // 3)
    fibonacci_to_take = max(1, required_numbers // 4)
    empirical_to_take = max(1, required_numbers // 4)

    # Add most frequent pairs
    top_pairs = sorted(pair_frequencies.items(), key=lambda x: x[1], reverse=True)[:pairs_to_take]
    for pair, _ in top_pairs:
        predicted_numbers.update(pair)

    # Add numbers from Fibonacci patterns
    if fibonacci_candidates:
        for num in fibonacci_candidates[:fibonacci_to_take]:
            predicted_numbers.add(num)
            if len(predicted_numbers) >= required_numbers:
                break

    # Add numbers from empirical patterns
    if empirical_patterns:
        for num in empirical_patterns[:empirical_to_take]:
            predicted_numbers.add(num)
            if len(predicted_numbers) >= required_numbers:
                break

    # Fill remaining slots with weighted random selection until we have exactly required_numbers
    while len(predicted_numbers) < required_numbers:
        # Calculate weights based on historical frequency
        number_weights = calculate_number_weights(
            past_draws,
            lottery_type_instance.min_number,
            lottery_type_instance.max_number
        )

        # Select remaining numbers using weighted random selection
        available_numbers = set(range(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number + 1
        )) - predicted_numbers

        if available_numbers:
            weights = [number_weights.get(num, 1) for num in available_numbers]
            selected = random.choices(list(available_numbers), weights=weights, k=1)[0]
            predicted_numbers.add(selected)

    # Ensure we return exactly required_numbers
    result = sorted(list(predicted_numbers))
    while len(result) > required_numbers:
        result.pop()

    return result


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