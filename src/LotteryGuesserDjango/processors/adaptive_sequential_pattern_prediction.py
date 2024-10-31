#adaptive_sequential_pattern_prediction.py
import random
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
import numpy as np
from itertools import combinations

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Adaptive algorithm that combines multiple pattern recognition approaches
    with dynamic weighting based on recent success patterns.
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
    """Generate a set of numbers based on adaptive sequential pattern prediction."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:200])

    past_numbers = [draw.lottery_type_number for draw in past_draws if isinstance(draw.lottery_type_number, list)]
    if not past_numbers:
        return generate_random_numbers(min_num, max_num, required_numbers)

    prediction_pool = set()

    # 1. Analyze sequential patterns
    sequence_numbers = analyze_sequential_patterns(
        past_numbers, min_num, max_num
    )
    prediction_pool.update(sequence_numbers[:required_numbers // 3])

    # 2. Cross-draw correlations
    correlation_numbers = find_cross_draw_correlations(past_numbers)
    prediction_pool.update(correlation_numbers[:required_numbers // 3])

    # 3. Frequency-based hot zones
    hot_zones = calculate_hot_zones(past_numbers, min_num, max_num)
    prediction_pool.update(hot_zones[:required_numbers // 3])

    # 4. Cyclic patterns
    cyclic_numbers = identify_cyclic_patterns(past_numbers)
    prediction_pool.update(cyclic_numbers[:required_numbers // 3])

    # Fill remaining slots
    while len(prediction_pool) < required_numbers:
        weights = calculate_adaptive_weights(
            past_numbers, min_num, max_num, prediction_pool
        )
        available_numbers = set(range(min_num, max_num + 1)) - prediction_pool
        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
            prediction_pool.add(selected)

    return sorted(prediction_pool)[:required_numbers]


# Analysis Functions
def analyze_sequential_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    sequence_scores = defaultdict(float)
    for draw in past_draws[:50]:  
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 2):
            for j in range(i + 1, len(sorted_draw) - 1):
                diff = sorted_draw[j] - sorted_draw[i]
                next_num = sorted_draw[j] + diff
                if min_num <= next_num <= max_num:
                    sequence_scores[next_num] += 1 / (past_draws.index(draw) + 1)
    return sorted(sequence_scores.keys(), key=sequence_scores.get, reverse=True)


def find_cross_draw_correlations(past_draws: List[List[int]]) -> List[int]:
    correlation_scores = defaultdict(float)
    for i in range(len(past_draws) - 1):
        current_draw = set(past_draws[i])
        next_draw = set(past_draws[i + 1])
        for num in current_draw & next_draw:
            correlation_scores[num] += 1
        for num in next_draw:
            for pair in combinations(current_draw, 2):
                if num > max(pair):
                    correlation_scores[num] += 0.2
    return sorted(correlation_scores.keys(), key=correlation_scores.get, reverse=True)


def calculate_hot_zones(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    frequency_scores = defaultdict(float)
    for i, draw in enumerate(past_draws):
        recency_weight = 1 / (i + 1)
        for num in draw:
            frequency_scores[num] += recency_weight
    range_size = (max_num - min_num) // 5
    for base in range(min_num, max_num - range_size + 1):
        zone_sum = sum(frequency_scores[n] for n in range(base, base + range_size))
        for num in range(base, base + range_size):
            frequency_scores[num] += zone_sum * 0.1
    return sorted(frequency_scores.keys(), key=frequency_scores.get, reverse=True)


def identify_cyclic_patterns(past_draws: List[List[int]]) -> List[int]:
    cycle_scores = defaultdict(float)
    if len(past_draws) >= 14:
        for i in range(len(past_draws) - 7):
            current_draw = set(past_draws[i])
            week_later_draw = set(past_draws[i + 7])
            for num in current_draw & week_later_draw:
                cycle_scores[num] += 1
    if len(past_draws) >= 28:
        for i in range(len(past_draws) - 28):
            current_draw = set(past_draws[i])
            month_later_draw = set(past_draws[i + 28])
            for num in current_draw & month_later_draw:
                cycle_scores[num] += 0.5
    return sorted(cycle_scores.keys(), key=cycle_scores.get, reverse=True)


def calculate_adaptive_weights(
        past_draws: List[List[int]], min_num: int, max_num: int, excluded_numbers: Set[int]
) -> Dict[int, float]:
    weights = defaultdict(float)
    all_numbers = [num for draw in past_draws for num in draw]
    frequency_counter = Counter(all_numbers)
    mean = statistics.mean(all_numbers)
    std_dev = statistics.stdev(all_numbers)
    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue
        weights[num] = frequency_counter.get(num, 0) / len(past_draws)
        z_score = abs((num - mean) / std_dev)
        weights[num] *= np.exp(-z_score / 2)
        last_seen = float('inf')
        for i, draw in enumerate(past_draws):
            if num in draw:
                last_seen = i
                break
        if last_seen != float('inf'):
            weights[num] *= np.exp(-last_seen / 10)
    return weights


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(numbers)
