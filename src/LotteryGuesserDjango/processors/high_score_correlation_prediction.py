# high_score_correlation_prediction.py
# Combines the best performing algorithms' approaches
# Focus on fibonacci, correlations, pair analysis and empirical decomposition

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from itertools import combinations, permutations
import random
from scipy.signal import find_peaks


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    High score hybrid algorithm that combines the most successful approaches
    based on historical performance metrics.
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:200])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    candidates = set()

    # 1. Fibonacci-based Analysis (40.0 score)
    fibonacci_numbers = analyze_fibonacci_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(fibonacci_numbers[:required_numbers // 4])

    # 2. Cross-Draw Correlation (35.0 score)
    correlation_numbers = analyze_cross_correlations(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(correlation_numbers[:required_numbers // 4])

    # 3. Number Pair Analysis (25.0 score)
    pair_numbers = analyze_number_pairs(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(pair_numbers[:required_numbers // 4])

    # Fill remaining slots using weighted combination
    while len(candidates) < required_numbers:
        weights = calculate_hybrid_weights(
            past_draws,
            lottery_type_instance.min_number,
            lottery_type_instance.max_number,
            candidates
        )

        available_numbers = set(range(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number + 1
        )) - candidates

        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
            candidates.add(selected)

    return sorted(list(candidates))[:required_numbers]


def analyze_fibonacci_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Enhanced Fibonacci pattern analysis."""
    fibonacci_scores = defaultdict(float)

    # Generate Fibonacci sequence up to max_num
    fibs = [1, 1]
    while fibs[-1] <= max_num:
        fibs.append(fibs[-1] + fibs[-2])

    # Analyze each draw for Fibonacci-like patterns
    for draw in past_draws:
        sorted_nums = sorted(draw)

        # Look for direct Fibonacci numbers
        for num in sorted_nums:
            if num in fibs:
                next_fib = num * 1.618033988749895  # Golden ratio
                if min_num <= next_fib <= max_num:
                    fibonacci_scores[int(next_fib)] += 1

        # Look for Fibonacci-like sequences
        for i in range(len(sorted_nums) - 2):
            if sorted_nums[i + 2] == sorted_nums[i + 1] + sorted_nums[i]:
                next_num = sorted_nums[i + 2] + sorted_nums[i + 1]
                if min_num <= next_num <= max_num:
                    fibonacci_scores[next_num] += 2

        # Analyze ratios close to golden ratio
        for i in range(len(sorted_nums) - 1):
            ratio = sorted_nums[i + 1] / sorted_nums[i] if sorted_nums[i] != 0 else 0
            if 1.5 <= ratio <= 1.7:  # Close to golden ratio
                next_num = int(sorted_nums[i + 1] * ratio)
                if min_num <= next_num <= max_num:
                    fibonacci_scores[next_num] += 1.5

    return sorted(fibonacci_scores.keys(), key=fibonacci_scores.get, reverse=True)


def analyze_cross_correlations(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Enhanced cross-draw correlation analysis."""
    correlation_scores = defaultdict(float)

    # Create sliding windows of draws
    window_sizes = [2, 3, 4]

    for size in window_sizes:
        for i in range(len(past_draws) - size):
            window = past_draws[i:i + size]

            # Analyze number transitions
            all_numbers = set()
            for draw in window:
                all_numbers.update(draw)

            # Calculate correlation strength
            for num1, num2 in combinations(sorted(all_numbers), 2):
                appearances = sum(1 for draw in window if num1 in draw and num2 in draw)
                if appearances >= size // 2:
                    # Project correlated numbers
                    diff = num2 - num1
                    next_num = num2 + diff
                    if min_num <= next_num <= max_num:
                        correlation_scores[next_num] += appearances / size

    # Analyze position-based correlations
    for i in range(len(past_draws) - 1):
        current = sorted(past_draws[i])
        next_draw = sorted(past_draws[i + 1])

        for pos in range(min(len(current), len(next_draw))):
            if pos < len(current) and pos < len(next_draw):
                diff = next_draw[pos] - current[pos]
                projected = next_draw[pos] + diff
                if min_num <= projected <= max_num:
                    correlation_scores[projected] += 1.5 / (i + 1)

    return sorted(correlation_scores.keys(), key=correlation_scores.get, reverse=True)


def analyze_number_pairs(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Enhanced number pair and affinity analysis."""
    pair_scores = defaultdict(float)

    # Analyze successful pairs
    pair_frequency = Counter()
    for draw in past_draws:
        sorted_nums = sorted(draw)
        for pair in combinations(sorted_nums, 2):
            pair_frequency[pair] += 1

    # Score based on pair patterns
    for (num1, num2), freq in pair_frequency.items():
        # Basic pair projection
        diff = num2 - num1
        next_num = num2 + diff
        if min_num <= next_num <= max_num:
            pair_scores[next_num] += freq

        # Analyze pair gaps
        gap = num2 - num1
        for base in [num1, num2]:
            for multiplier in [0.5, 1.0, 1.5, 2.0]:
                projected = int(base + gap * multiplier)
                if min_num <= projected <= max_num:
                    pair_scores[projected] += freq * (1 / multiplier)

    # Analyze pair affinities
    for i in range(len(past_draws) - 1):
        current_pairs = set(combinations(sorted(past_draws[i]), 2))
        next_pairs = set(combinations(sorted(past_draws[i + 1]), 2))

        # Find pairs that tend to appear together
        common_pairs = current_pairs & next_pairs
        for pair in common_pairs:
            # Project based on common pair patterns
            avg = sum(pair) / 2
            diff = pair[1] - pair[0]

            for offset in [-1, 1]:
                projected = int(avg + offset * diff)
                if min_num <= projected <= max_num:
                    pair_scores[projected] += 2 / (i + 1)

    return sorted(pair_scores.keys(), key=pair_scores.get, reverse=True)


def calculate_hybrid_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate weights using combination of successful approaches."""
    weights = defaultdict(float)

    if not past_draws:
        return weights

    # Calculate base frequency weights
    frequency = Counter(num for draw in past_draws for num in draw)
    total_numbers = sum(frequency.values())

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base frequency weight
        weights[num] = frequency.get(num, 0) / total_numbers

        # Fibonacci weight
        closest_fib_ratio = float('inf')
        fib_prev, fib_curr = 1, 1
        while fib_curr <= max_num:
            if abs(num - fib_curr) < abs(closest_fib_ratio):
                closest_fib_ratio = num - fib_curr
            fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
        weights[num] *= (1 + 1 / (1 + abs(closest_fib_ratio)))

        # Correlation weight
        last_draw = past_draws[0] if past_draws else []
        if last_draw:
            correlations = sum(1 / (1 + abs(num - n)) for n in last_draw)
            weights[num] *= (1 + correlations / len(last_draw))

        # Pair affinity weight
        if len(past_draws) >= 2:
            current_pairs = set(combinations(sorted(past_draws[0]), 2))
            prev_pairs = set(combinations(sorted(past_draws[1]), 2))
            pair_strength = len(current_pairs & prev_pairs) / len(current_pairs) if current_pairs else 0
            weights[num] *= (1 + pair_strength)

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers using hybrid patterns."""
    numbers = set()
    required_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Initialize with Fibonacci-like spacing
    golden_ratio = 1.618033988749895
    current = lottery_type_instance.min_number

    while len(numbers) < required_numbers and current <= lottery_type_instance.max_number:
        if current >= lottery_type_instance.min_number:
            numbers.add(int(current))
        current *= golden_ratio

    # Fill remaining with balanced random selection
    while len(numbers) < required_numbers:
        num = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        numbers.add(num)

    return sorted(list(numbers))[:required_numbers]