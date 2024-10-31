# deep_pattern_decomposition_prediction.py
# Combines deep pattern recognition with hierarchical decomposition
# and multivariate statistical analysis for lottery number prediction

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy.stats import entropy, mode
from sklearn.preprocessing import MinMaxScaler
import random
from itertools import combinations, permutations
import math


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deep pattern decomposition algorithm for combined lottery types.
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


def get_pattern_statistics(past_draws: List[List[int]]) -> Dict[str, Dict]:
    """
    Get comprehensive pattern statistics.

    Returns a dictionary containing:
    - position_stats
    - sequence_stats
    - distance_stats
    - entropy_stats
    """
    if not past_draws:
        return {}

    stats = {
        'position_stats': analyze_position_statistics(past_draws),
        'sequence_stats': analyze_sequence_statistics(past_draws),
        'distance_stats': analyze_distance_statistics(past_draws),
        'entropy_stats': analyze_entropy_statistics(past_draws)
    }

    return stats

def select_next_number(
        probabilities: Dict[int, float],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> int:
    """Select next number based on probabilities."""
    available_numbers = [
        num for num in range(min_num, max_num + 1)
        if num not in excluded_numbers
    ]

    if not available_numbers:
        return random.randint(min_num, max_num)

    weights = [probabilities.get(num, 0.1) for num in available_numbers]
    return random.choices(available_numbers, weights=weights, k=1)[0]


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))

def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using deep pattern analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    candidates = set()

    # 1. Deep Pattern Analysis
    pattern_numbers = analyze_deep_patterns(past_draws, min_num, max_num)
    candidates.update(pattern_numbers[:required_numbers // 3])

    # 2. Statistical Decomposition
    decomp_numbers = perform_statistical_decomposition(past_draws, min_num, max_num)
    candidates.update(decomp_numbers[:required_numbers // 3])

    # 3. Entropy Analysis
    entropy_numbers = analyze_entropy_patterns(past_draws, min_num, max_num)
    candidates.update(entropy_numbers[:required_numbers // 3])

    # Fill remaining using composite probabilities
    while len(candidates) < required_numbers:
        probabilities = calculate_composite_probabilities(
            past_draws,
            min_num,
            max_num,
            candidates
        )
        candidates.add(select_next_number(probabilities, min_num, max_num, candidates))

    return sorted(list(candidates))[:required_numbers]


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:300])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def analyze_deep_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Deep pattern analysis using hierarchical decomposition."""
    pattern_scores = defaultdict(float)

    # Analyze position patterns
    position_patterns = analyze_position_patterns(past_draws)
    for num, score in position_patterns.items():
        if min_num <= num <= max_num:
            pattern_scores[num] += score * 0.4

    # Analyze sequence patterns
    sequence_patterns = analyze_sequence_patterns(past_draws)
    for num, score in sequence_patterns.items():
        if min_num <= num <= max_num:
            pattern_scores[num] += score * 0.3

    # Analyze distance patterns
    distance_patterns = analyze_distance_patterns(past_draws)
    for num, score in distance_patterns.items():
        if min_num <= num <= max_num:
            pattern_scores[num] += score * 0.3

    return sorted(pattern_scores.keys(), key=pattern_scores.get, reverse=True)


def analyze_position_patterns(past_draws: List[List[int]]) -> Dict[int, float]:
    """Analyze patterns based on number positions."""
    position_scores = defaultdict(float)

    # Create position-based probability matrix
    num_positions = max(len(draw) for draw in past_draws)
    position_matrix = defaultdict(lambda: defaultdict(int))

    for draw in past_draws:
        sorted_draw = sorted(draw)
        for pos, num in enumerate(sorted_draw):
            position_matrix[pos][num] += 1

            # Look for position transitions
            if pos > 0:
                prev_num = sorted_draw[pos - 1]
                delta = num - prev_num
                position_scores[prev_num + delta] += 0.1

    # Calculate position probabilities
    total_draws = len(past_draws)
    for pos in range(num_positions):
        pos_total = sum(position_matrix[pos].values())
        if pos_total > 0:
            for num, count in position_matrix[pos].items():
                prob = count / pos_total
                position_scores[num] += prob * (1 + 0.1 * pos)  # Higher weight for later positions

    # Analyze position transitions
    for i in range(len(past_draws) - 1):
        current = past_draws[i]
        next_draw = past_draws[i + 1]

        for pos in range(min(len(current), len(next_draw))):
            if pos < len(current) and pos < len(next_draw):
                transition = next_draw[pos] - current[pos]
                predicted = next_draw[pos] + transition
                position_scores[predicted] += 0.2 / (i + 1)

    return position_scores


def analyze_sequence_patterns(past_draws: List[List[int]]) -> Dict[int, float]:
    """Analyze sequential patterns in the data."""
    sequence_scores = defaultdict(float)

    # Analyze arithmetic sequences
    for draw in past_draws:
        sorted_nums = sorted(draw)
        for i in range(len(sorted_nums) - 2):
            for j in range(i + 1, len(sorted_nums) - 1):
                diff = sorted_nums[j] - sorted_nums[i]
                next_num = sorted_nums[j] + diff
                sequence_scores[next_num] += 1 / (past_draws.index(draw) + 1)

    # Analyze geometric sequences
    for draw in past_draws:
        sorted_nums = sorted(draw)
        for i in range(len(sorted_nums) - 2):
            for j in range(i + 1, len(sorted_nums) - 1):
                if sorted_nums[i] != 0:
                    ratio = sorted_nums[j] / sorted_nums[i]
                    next_num = int(sorted_nums[j] * ratio)
                    sequence_scores[next_num] += 0.5 / (past_draws.index(draw) + 1)

    # Look for Fibonacci-like sequences
    for draw in past_draws[:50]:  # Focus on recent draws
        sorted_nums = sorted(draw)
        for i in range(len(sorted_nums) - 2):
            sum_next = sorted_nums[i] + sorted_nums[i + 1]
            if sum_next in sequence_scores:
                sequence_scores[sum_next] += 0.3

    return sequence_scores


def analyze_distance_patterns(past_draws: List[List[int]]) -> Dict[int, float]:
    """Analyze patterns in the distances between numbers."""
    distance_scores = defaultdict(float)

    # Analyze consecutive number distances
    for draw in past_draws:
        sorted_nums = sorted(draw)
        distances = [sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]

        if distances:
            avg_distance = sum(distances) / len(distances)
            next_num = sorted_nums[-1] + avg_distance
            distance_scores[int(next_num)] += 1

    # Analyze gap patterns
    for i in range(len(past_draws) - 1):
        current = set(past_draws[i])
        next_draw = set(past_draws[i + 1])

        # Find numbers that maintain similar gaps
        gaps = [n2 - n1 for n1, n2 in combinations(sorted(current), 2)]
        for gap in gaps:
            for num in current:
                predicted = num + gap
                if predicted in next_draw:
                    distance_scores[predicted + gap] += 0.5

    # Look for symmetric patterns
    for draw in past_draws:
        sorted_nums = sorted(draw)
        mid = len(sorted_nums) // 2

        for i in range(mid):
            diff = sorted_nums[-(i + 1)] - sorted_nums[i]
            next_symmetric = sorted_nums[-(i + 1)] + diff
            distance_scores[next_symmetric] += 0.3

    return distance_scores


def perform_statistical_decomposition(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Perform statistical decomposition of the number patterns."""
    decomp_scores = defaultdict(float)

    # Create number frequency matrix
    number_freq = Counter(num for draw in past_draws for num in draw)
    total_numbers = sum(number_freq.values())

    # Basic probability scores
    for num in range(min_num, max_num + 1):
        prob = number_freq.get(num, 0) / total_numbers
        decomp_scores[num] += prob

    # Analyze statistical moments
    all_numbers = [num for draw in past_draws for num in draw]
    if all_numbers:
        mean = statistics.mean(all_numbers)
        std_dev = statistics.stdev(all_numbers)

        for num in range(min_num, max_num + 1):
            # Score based on distance from mean
            z_score = abs((num - mean) / std_dev)
            decomp_scores[num] += np.exp(-z_score / 2)

    # Analyze distribution characteristics
    for num in range(min_num, max_num + 1):
        # Check for mode proximity
        try:
            mode_val = mode(all_numbers).mode[0]
            mode_distance = abs(num - mode_val)
            decomp_scores[num] += 1 / (1 + mode_distance)
        except:
            pass

    return sorted(decomp_scores.keys(), key=decomp_scores.get, reverse=True)


def analyze_entropy_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze entropy-based patterns in the number sequence."""
    entropy_scores = defaultdict(float)

    # Calculate base probabilities
    number_freq = Counter(num for draw in past_draws for num in draw)
    total_numbers = sum(number_freq.values())
    probabilities = {num: count / total_numbers for num, count in number_freq.items()}

    # Calculate system entropy
    probs = list(probabilities.values())
    system_entropy = entropy(probs) if probs else 0

    # Score numbers based on their contribution to entropy
    for num in range(min_num, max_num + 1):
        # Calculate entropy change if this number is added
        new_probs = probabilities.copy()
        new_probs[num] = new_probs.get(num, 0) + 1 / (total_numbers + 1)
        new_entropy = entropy(list(new_probs.values()))

        # Score based on entropy difference
        entropy_scores[num] += abs(new_entropy - system_entropy)

    return sorted(entropy_scores.keys(), key=entropy_scores.get, reverse=True)


def calculate_composite_probabilities(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate composite probabilities for number selection."""
    probabilities = defaultdict(float)

    # Base frequency probabilities
    number_freq = Counter(num for draw in past_draws for num in draw)
    total_numbers = sum(number_freq.values())

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Basic probability from frequency
        base_prob = number_freq.get(num, 0) / total_numbers

        # Recent appearance factor
        recency = 0
        for i, draw in enumerate(past_draws):
            if num in draw:
                recency = 1 / (i + 1)
                break

        # Position bias
        position_score = 0
        for draw in past_draws[:20]:  # Focus on recent draws
            if num in draw:
                pos = sorted(draw).index(num)
                position_score += 1 / (pos + 1)

        # Combine factors
        probabilities[num] = (base_prob + recency + position_score / 20) / 3

    # Normalize probabilities
    total_prob = sum(probabilities.values())
    if total_prob > 0:
        for num in probabilities:
            probabilities[num] /= total_prob

    return probabilities


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        numbers.add(random.randint(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number
        ))
    return sorted(list(numbers))