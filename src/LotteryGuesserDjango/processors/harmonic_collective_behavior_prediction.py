# harmonic_collective_behavior_prediction.py
# Advanced harmonic pattern analysis with collective behavior modeling
# Combines musical harmony theory with swarm intelligence principles

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy.signal import find_peaks
import math
from itertools import combinations, chain
import random


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Harmonic pattern analyzer for combined lottery types."""
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
    """Generate numbers using harmonic analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    candidates = set()

    # 1. Harmonic Analysis
    harmonic_numbers = analyze_harmonic_patterns(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(harmonic_numbers[:required_numbers // 3])

    # 2. Collective Behavior
    collective_numbers = analyze_collective_behavior(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(collective_numbers[:required_numbers // 3])

    # 3. Rhythm Analysis
    rhythm_numbers = analyze_rhythm_patterns(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(rhythm_numbers[:required_numbers // 3])

    # Fill remaining slots
    while len(candidates) < required_numbers:
        weights = calculate_harmonic_probabilities(
            past_draws,
            min_num,
            max_num,
            candidates
        )

        available_numbers = set(range(min_num, max_num + 1)) - candidates
        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(
                list(available_numbers),
                weights=number_weights,
                k=1
            )[0]
            candidates.add(selected)

    return sorted(list(candidates))[:required_numbers]


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:200])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def analyze_harmonic_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze harmonic relationships between numbers using music theory principles."""
    harmonic_scores = defaultdict(float)

    # Musical intervals (in ratio form)
    harmonic_ratios = {
        'octave': 2 / 1,
        'perfect_fifth': 3 / 2,
        'perfect_fourth': 4 / 3,
        'major_third': 5 / 4,
        'minor_third': 6 / 5,
        'major_sixth': 5 / 3,
        'golden_ratio': 1.618033988749895
    }

    # Analyze each draw for harmonic relationships
    for draw in past_draws:
        sorted_nums = sorted(draw)

        # Find numbers that form harmonic ratios
        for i, num1 in enumerate(sorted_nums):
            if num1 == 0:
                continue

            for j in range(i + 1, len(sorted_nums)):
                ratio = sorted_nums[j] / num1

                # Check against harmonic ratios
                for interval, target_ratio in harmonic_ratios.items():
                    if abs(ratio - target_ratio) < 0.1:
                        # Project next harmonic number
                        next_harmonic = int(sorted_nums[j] * target_ratio)
                        if min_num <= next_harmonic <= max_num:
                            harmonic_scores[next_harmonic] += 1

                        # Also consider the reverse progression
                        prev_harmonic = int(num1 / target_ratio)
                        if min_num <= prev_harmonic <= max_num:
                            harmonic_scores[prev_harmonic] += 0.5

    # Analyze compound harmonics
    for draw in past_draws[:50]:  # Focus on recent draws
        sorted_nums = sorted(draw)

        # Find numbers that form compound ratios
        for i in range(len(sorted_nums) - 2):
            if sorted_nums[i] == 0:
                continue

            ratio1 = sorted_nums[i + 1] / sorted_nums[i]
            ratio2 = sorted_nums[i + 2] / sorted_nums[i + 1]

            # Project next number in compound progression
            next_number = int(sorted_nums[i + 2] * (ratio1 + ratio2) / 2)
            if min_num <= next_number <= max_num:
                harmonic_scores[next_number] += 0.8

    return sorted(harmonic_scores.keys(), key=harmonic_scores.get, reverse=True)


def analyze_collective_behavior(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze collective behavior patterns using swarm intelligence principles."""
    collective_scores = defaultdict(float)

    # Create virtual agents for each number
    agents = defaultdict(lambda: {
        'position': 0,
        'velocity': 0,
        'best_position': 0,
        'success_rate': 0
    })

    # Initialize agents with historical data
    for draw in past_draws:
        for num in draw:
            if num not in agents:
                agents[num] = {
                    'position': num,
                    'velocity': 0,
                    'best_position': num,
                    'success_rate': 1
                }
            else:
                agents[num]['success_rate'] += 1

    # Normalize success rates
    max_success = max(agent['success_rate'] for agent in agents.values())
    if max_success > 0:
        for agent in agents.values():
            agent['success_rate'] /= max_success

    # Simulate collective behavior
    for i in range(10):  # Multiple iterations
        for num, agent in agents.items():
            # Get neighboring agents
            neighbors = [
                n for n in range(max(min_num, num - 5), min(max_num, num + 6))
                if n in agents and n != num
            ]

            if neighbors:
                # Calculate center of mass
                center = sum(agents[n]['position'] for n in neighbors) / len(neighbors)

                # Update velocity using swarm rules
                cohesion = (center - agent['position']) * 0.1
                separation = sum(
                    (agent['position'] - agents[n]['position'])
                    for n in neighbors if abs(agent['position'] - agents[n]['position']) < 3
                ) * 0.05

                # Update position
                agent['velocity'] = agent['velocity'] * 0.9 + cohesion + separation
                new_position = int(agent['position'] + agent['velocity'])

                # Score positions near successful agents
                if min_num <= new_position <= max_num:
                    collective_scores[new_position] += agent['success_rate']

    # Analyze group formations
    for i in range(len(past_draws) - 1):
        current_group = set(past_draws[i])
        next_group = set(past_draws[i + 1])

        # Find stable group patterns
        stable_numbers = current_group & next_group
        for num in stable_numbers:
            # Project group movement
            movement = sum(n - num for n in next_group) / len(next_group)
            predicted = int(num + movement)
            if min_num <= predicted <= max_num:
                collective_scores[predicted] += 1 / (i + 1)

    return sorted(collective_scores.keys(), key=collective_scores.get, reverse=True)


def analyze_rhythm_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze rhythmic patterns in the number sequence."""
    rhythm_scores = defaultdict(float)

    # Convert numbers to intervals
    all_intervals = []
    for draw in past_draws:
        sorted_nums = sorted(draw)
        intervals = [sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
        all_intervals.append(intervals)

    # Find common rhythmic patterns
    pattern_length = 3  # Look for patterns of 3 intervals
    rhythm_patterns = defaultdict(int)

    for intervals in all_intervals:
        if len(intervals) >= pattern_length:
            for i in range(len(intervals) - pattern_length + 1):
                pattern = tuple(intervals[i:i + pattern_length])
                rhythm_patterns[pattern] += 1

    # Use rhythmic patterns to predict next numbers
    for draw in past_draws[:30]:  # Focus on recent draws
        sorted_nums = sorted(draw)
        current_intervals = [sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]

        if len(current_intervals) >= pattern_length - 1:
            # Look for matching rhythm patterns
            current_pattern = tuple(current_intervals[-(pattern_length - 1):])

            for pattern, count in rhythm_patterns.items():
                if pattern[:-1] == current_pattern:
                    # Project next number based on rhythm
                    next_number = sorted_nums[-1] + pattern[-1]
                    if min_num <= next_number <= max_num:
                        rhythm_scores[next_number] += count

    # Analyze polyrhythms (multiple simultaneous rhythms)
    for i in range(len(past_draws) - 2):
        intervals1 = [past_draws[i + 1][j] - past_draws[i][j] for j in range(len(past_draws[i]))]
        intervals2 = [past_draws[i + 2][j] - past_draws[i + 1][j] for j in range(len(past_draws[i + 1]))]

        if intervals1 and intervals2:
            # Find cross-rhythm patterns
            avg_interval1 = sum(intervals1) / len(intervals1)
            avg_interval2 = sum(intervals2) / len(intervals2)

            # Project next numbers based on polyrhythm
            for num in past_draws[i + 2]:
                next_num1 = int(num + avg_interval1)
                next_num2 = int(num + avg_interval2)

                if min_num <= next_num1 <= max_num:
                    rhythm_scores[next_num1] += 0.5
                if min_num <= next_num2 <= max_num:
                    rhythm_scores[next_num2] += 0.5

    return sorted(rhythm_scores.keys(), key=rhythm_scores.get, reverse=True)


def calculate_harmonic_probabilities(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate probabilities based on harmonic principles."""
    probabilities = defaultdict(float)

    # Calculate base frequencies
    number_freq = Counter(num for draw in past_draws for num in draw)
    total_numbers = sum(number_freq.values())

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base probability from frequency
        base_prob = number_freq.get(num, 0) / total_numbers

        # Harmonic resonance
        resonance_score = 0
        for draw in past_draws[:20]:  # Focus on recent draws
            for existing_num in draw:
                if existing_num != 0:
                    ratio = num / existing_num
                    # Check for harmonic ratios
                    harmonic_factor = min(
                        abs(ratio - r)
                        for r in [2 / 1, 3 / 2, 4 / 3, 5 / 3, 5 / 4, 6 / 5, 1.618033988749895]
                    )
                    resonance_score += 1 / (1 + harmonic_factor)

        # Rhythmic stability
        rhythm_score = 0
        for i in range(len(past_draws) - 1):
            if num in past_draws[i] and num in past_draws[i + 1]:
                rhythm_score += 1 / (i + 1)

        # Combine scores
        probabilities[num] = (base_prob + resonance_score / 20 + rhythm_score) / 3

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