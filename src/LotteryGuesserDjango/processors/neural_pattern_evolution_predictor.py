#neural_pattern_evolution_predictor.py
import random
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
import numpy as np
from itertools import combinations
from scipy.stats import zscore
import math


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Neural pattern evolution algorithm that combines multiple advanced approaches
    with dynamic pattern recognition and evolutionary optimization.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate a set of numbers using the neural pattern evolution algorithm."""
    # Get extended historical data for better pattern recognition
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:500])

    # Extract appropriate number sets from past draws
    if is_main:
        past_numbers = [draw.lottery_type_number for draw in past_draws
                        if isinstance(draw.lottery_type_number, list)]
    else:
        past_numbers = [draw.additional_numbers for draw in past_draws
                        if hasattr(draw, 'additional_numbers') and
                        isinstance(draw.additional_numbers, list)]

    if not past_numbers:
        return generate_random_numbers(min_num, max_num, required_numbers)

    candidates = set()

    # 1. Neural Pattern Analysis
    neural_patterns = analyze_neural_patterns(
        past_numbers,
        min_num,
        max_num
    )
    candidates.update(neural_patterns[:required_numbers // 2])

    # 2. Evolutionary Distance Patterns
    evolution_numbers = find_evolutionary_distances(
        past_numbers,
        min_num,
        max_num
    )
    candidates.update(evolution_numbers[:required_numbers // 2])

    # 3. Quantum-inspired Pattern Detection
    quantum_patterns = detect_quantum_patterns(
        past_numbers,
        min_num,
        max_num
    )
    candidates.update(quantum_patterns[:required_numbers // 2])

    # 4. Apply Matrix Transformation Patterns
    matrix_patterns = analyze_matrix_patterns(past_numbers)
    candidates.update(matrix_patterns[:required_numbers // 2])

    # Fill remaining slots with neural weighted selection
    while len(candidates) < required_numbers:
        weights = calculate_neural_weights(
            past_numbers,
            min_num,
            max_num,
            candidates
        )

        available_numbers = set(range(min_num, max_num + 1)) - candidates

        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
            candidates.add(selected)

    return sorted(list(candidates))[:required_numbers]


def analyze_neural_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Advanced neural pattern analysis using temporal convolution."""
    pattern_scores = defaultdict(float)

    # Time window analysis with exponential decay
    for time_window in [5, 10, 20, 50]:
        recent_draws = past_draws[:time_window]
        decay_factor = np.exp(-0.1 * time_window)

        for i, draw in enumerate(recent_draws):
            draw_weight = decay_factor * np.exp(-0.1 * i)

            # Analyze number relationships
            sorted_draw = sorted(draw)
            for j in range(len(sorted_draw) - 1):
                diff = sorted_draw[j + 1] - sorted_draw[j]
                next_potential = sorted_draw[j + 1] + diff
                if min_num <= next_potential <= max_num:
                    pattern_scores[next_potential] += draw_weight

                # Analyze multiplicative relationships
                if sorted_draw[j] != 0:
                    ratio = sorted_draw[j + 1] / sorted_draw[j]
                    next_ratio = int(sorted_draw[j + 1] * ratio)
                    if min_num <= next_ratio <= max_num:
                        pattern_scores[next_ratio] += draw_weight * 0.5

    return sorted(pattern_scores.keys(), key=pattern_scores.get, reverse=True)


def find_evolutionary_distances(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze evolutionary distances between numbers using dynamic programming."""
    distance_patterns = defaultdict(float)

    # Create distance matrices
    for i in range(len(past_draws) - 1):
        current = set(past_draws[i])
        next_draw = set(past_draws[i + 1])

        # Calculate evolutionary distances
        for num1 in current:
            for num2 in next_draw:
                distance = abs(num2 - num1)
                if min_num <= distance <= max_num:
                    distance_patterns[distance] += 1 / (i + 1)

                # Golden ratio analysis
                golden_next = int(num1 * 1.618033988749895)
                if min_num <= golden_next <= max_num:
                    distance_patterns[golden_next] += 0.5 / (i + 1)

    return sorted(distance_patterns.keys(), key=distance_patterns.get, reverse=True)


def detect_quantum_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Quantum-inspired pattern detection using superposition principles."""
    quantum_scores = defaultdict(float)

    # Create quantum-inspired state vectors
    for i, draw in enumerate(past_draws[:50]):
        # Phase analysis
        phase_factor = np.exp(-0.1 * i)
        sorted_draw = sorted(draw)

        # Quantum walks
        for step in range(1, len(sorted_draw)):
            for j in range(len(sorted_draw) - step):
                current = sorted_draw[j]
                next_pos = sorted_draw[j + step]

                # Quantum tunneling effect
                tunnel_positions = range(current, next_pos + 1)
                for pos in tunnel_positions:
                    if min_num <= pos <= max_num:
                        quantum_scores[pos] += phase_factor / (step + 1)

                # Quantum interference patterns
                interference = (current + next_pos) // 2
                if min_num <= interference <= max_num:
                    quantum_scores[interference] += phase_factor * 1.5

    return sorted(quantum_scores.keys(), key=quantum_scores.get, reverse=True)


def analyze_matrix_patterns(past_draws: List[List[int]]) -> List[int]:
    """Analyze patterns using matrix transformations."""
    matrix_scores = defaultdict(float)

    if not past_draws:
        return []

    # Create sliding windows of draws
    window_size = min(10, len(past_draws))
    for i in range(len(past_draws) - window_size + 1):
        window = past_draws[i:i + window_size]

        # Create matrix from window
        # Ensure all draws in window have the same length by padding with zeros
        max_len = max(len(draw) for draw in window)
        padded_window = [draw + [0] * (max_len - len(draw)) for draw in window]
        matrix = np.array(padded_window)

        # Analyze column patterns
        for col in range(matrix.shape[1]):
            column_values = matrix[:, col]
            diff = np.diff(column_values)

            if len(diff) > 0:  # Check if there are enough values to calculate differences
                # Predict next value using various methods
                linear_next = int(column_values[-1] + np.mean(diff))

                # Only calculate exp_next if all differences are positive
                positive_diffs = diff[diff > 0]
                if len(positive_diffs) > 0:
                    exp_next = int(column_values[-1] * np.exp(np.mean(np.log(positive_diffs))))
                    matrix_scores[exp_next] += 0.8 / (i + 1)

                matrix_scores[linear_next] += 1 / (i + 1)

                # Analyze diagonal patterns
                if col < matrix.shape[1] - 1:
                    diag_values = np.diagonal(matrix[:, col:])
                    diag_diff = np.diff(diag_values)
                    if len(diag_diff) > 0:
                        diag_next = int(diag_values[-1] + np.mean(diag_diff))
                        matrix_scores[diag_next] += 1.2 / (i + 1)

    return sorted(matrix_scores.keys(), key=matrix_scores.get, reverse=True)


def calculate_neural_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate neural network inspired weights for number selection."""
    weights = defaultdict(float)

    if not past_draws:
        return weights

    # Basic frequency analysis
    all_numbers = [num for draw in past_draws for num in draw]

    if not all_numbers:
        return weights

    frequency_counter = Counter(all_numbers)

    # Statistical measures
    mean = statistics.mean(all_numbers)
    std_dev = statistics.stdev(all_numbers) if len(all_numbers) > 1 else 1

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base frequency weight
        base_weight = frequency_counter.get(num, 0) / len(past_draws)

        # Neural activation function (sigmoid)
        z_score = (num - mean) / std_dev
        activation = 1 / (1 + np.exp(-z_score))

        # Time since last appearance
        last_seen = float('inf')
        for i, draw in enumerate(past_draws):
            if num in draw:
                last_seen = i
                break

        # Combine weights with neural activation
        time_factor = np.exp(-last_seen / 20) if last_seen != float('inf') else 0.1
        weights[num] = base_weight * (0.4 + 0.6 * activation) * time_factor

        # Add positional bias
        positions = [i for draw in past_draws[:20] for i, x in enumerate(sorted(draw)) if x == num]
        if positions:
            pos_weight = np.mean(positions) / len(past_draws[0])
            weights[num] *= (0.5 + pos_weight)

    return weights


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))