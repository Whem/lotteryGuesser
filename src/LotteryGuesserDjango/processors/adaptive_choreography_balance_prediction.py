# adaptive_choreography_balance_prediction.py
# Dynamic balance analysis with choreographic pattern recognition
# Combines movement patterns with equilibrium seeking behavior

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy.spatial import distance
import random
from itertools import combinations
from sklearn.cluster import KMeans


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Adaptive choreography algorithm that combines movement patterns
    with dynamic equilibrium seeking behavior.
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:200])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    candidates = set()

    # 1. Movement Pattern Analysis
    movement_numbers = analyze_movement_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(movement_numbers[:required_numbers // 3])

    # 2. Balance Point Analysis
    balance_numbers = analyze_balance_points(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(balance_numbers[:required_numbers // 3])

    # 3. Choreographic Sequence Analysis
    sequence_numbers = analyze_choreographic_sequences(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(sequence_numbers[:required_numbers // 3])

    # Fill remaining slots using dynamic equilibrium
    while len(candidates) < required_numbers:
        weights = calculate_equilibrium_weights(
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


def analyze_movement_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze movement patterns in number sequences."""
    movement_scores = defaultdict(float)

    # Create movement vectors
    vectors = []
    for i in range(len(past_draws) - 1):
        current = sorted(past_draws[i])
        next_draw = sorted(past_draws[i + 1])

        # Calculate movement vectors between consecutive draws
        movements = []
        for j in range(min(len(current), len(next_draw))):
            movement = next_draw[j] - current[j]
            movements.append(movement)
        vectors.append(movements)

    if not vectors:
        return []

    # Analyze movement patterns using clustering
    try:
        # Pad vectors to same length
        max_len = max(len(v) for v in vectors)
        padded_vectors = [v + [0] * (max_len - len(v)) for v in vectors]

        # Cluster movement patterns
        kmeans = KMeans(n_clusters=min(5, len(padded_vectors)), n_init=10)
        clusters = kmeans.fit_predict(padded_vectors)

        # Score numbers based on cluster centers
        for center in kmeans.cluster_centers_:
            last_draw = sorted(past_draws[-1])
            for movement in center:
                for base in last_draw:
                    predicted = int(base + movement)
                    if min_num <= predicted <= max_num:
                        movement_scores[predicted] += 1

        # Analyze momentum
        for i in range(len(vectors)):
            momentum = np.mean(vectors[i]) if vectors[i] else 0
            weight = 1 / (i + 1)  # Recent movements have higher weight

            last_numbers = sorted(past_draws[-1])
            for base in last_numbers:
                predicted = int(base + momentum)
                if min_num <= predicted <= max_num:
                    movement_scores[predicted] += weight

    except Exception as e:
        print(f"Movement analysis error: {e}")

    return sorted(movement_scores.keys(), key=movement_scores.get, reverse=True)


def analyze_balance_points(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze balance points and equilibrium states."""
    balance_scores = defaultdict(float)

    # Calculate different types of balance points
    for draw in past_draws:
        sorted_nums = sorted(draw)

        # Center of mass
        com = sum(sorted_nums) / len(sorted_nums)
        balance_scores[int(com)] += 1

        # Median points
        if len(sorted_nums) % 2 == 0:
            median = (sorted_nums[len(sorted_nums) // 2 - 1] + sorted_nums[len(sorted_nums) // 2]) / 2
            balance_scores[int(median)] += 1
        else:
            median = sorted_nums[len(sorted_nums) // 2]
            balance_scores[median] += 1

        # Dynamic equilibrium points
        for i in range(len(sorted_nums) - 1):
            for j in range(i + 1, len(sorted_nums)):
                balance = (sorted_nums[i] + sorted_nums[j]) / 2
                if min_num <= balance <= max_num:
                    balance_scores[int(balance)] += 0.5 / (j - i)

    # Analyze stability regions
    stability_regions = find_stability_regions(past_draws, min_num, max_num)
    for region_center, stability in stability_regions.items():
        if min_num <= region_center <= max_num:
            balance_scores[region_center] += stability

    return sorted(balance_scores.keys(), key=balance_scores.get, reverse=True)


def analyze_choreographic_sequences(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze choreographic sequences in the number patterns."""
    sequence_scores = defaultdict(float)

    # Create positional transitions
    position_transitions = defaultdict(list)
    for i in range(len(past_draws) - 1):
        current = sorted(past_draws[i])
        next_draw = sorted(past_draws[i + 1])

        for pos in range(min(len(current), len(next_draw))):
            position_transitions[pos].append((current[pos], next_draw[pos]))

    # Analyze positional choreography
    for pos, transitions in position_transitions.items():
        # Calculate common movements
        movements = [end - start for start, end in transitions]
        if movements:
            avg_movement = sum(movements) / len(movements)
            recent_numbers = sorted(past_draws[-1]) if past_draws else []

            # Project movements
            if pos < len(recent_numbers):
                projected = int(recent_numbers[pos] + avg_movement)
                if min_num <= projected <= max_num:
                    sequence_scores[projected] += 1

    # Analyze spatial patterns
    spatial_patterns = find_spatial_patterns(past_draws)
    for pattern, weight in spatial_patterns.items():
        # Project pattern continuations
        continuations = project_pattern_continuations(pattern, min_num, max_num)
        for num in continuations:
            sequence_scores[num] += weight

    return sorted(sequence_scores.keys(), key=sequence_scores.get, reverse=True)


def find_stability_regions(past_draws: List[List[int]], min_num: int, max_num: int) -> Dict[int, float]:
    """Find stable regions in the number space."""
    stability = defaultdict(float)

    # Create number density map
    density = Counter(num for draw in past_draws for num in draw)

    # Find stable regions using sliding window
    window_size = (max_num - min_num) // 10
    for center in range(min_num + window_size // 2, max_num - window_size // 2):
        region_numbers = []
        for i in range(center - window_size // 2, center + window_size // 2):
            region_numbers.extend([i] * density.get(i, 0))

        if region_numbers:
            # Calculate stability metrics
            std_dev = statistics.stdev(region_numbers) if len(region_numbers) > 1 else float('inf')
            density_score = len(region_numbers) / (window_size * len(past_draws))

            # Combine metrics into stability score
            stability[center] = density_score * (1 / (1 + std_dev))

    return stability


def find_spatial_patterns(past_draws: List[List[int]]) -> Dict[Tuple[int, ...], float]:
    """Find recurring spatial patterns in the number sequences."""
    patterns = defaultdict(float)

    # Look for different pattern lengths
    for length in [2, 3, 4]:
        for draw in past_draws:
            sorted_nums = sorted(draw)

            # Extract sub-patterns
            for i in range(len(sorted_nums) - length + 1):
                pattern = tuple(sorted_nums[i:i + length])

                # Calculate pattern significance
                pattern_gaps = [pattern[j + 1] - pattern[j] for j in range(len(pattern) - 1)]
                regularity = len(set(pattern_gaps)) == 1  # Check if gaps are uniform

                weight = 1.0
                if regularity:
                    weight *= 2.0

                patterns[pattern] += weight / length

    return patterns


def project_pattern_continuations(pattern: Tuple[int, ...], min_num: int, max_num: int) -> List[int]:
    """Project possible continuations of a spatial pattern."""
    if len(pattern) < 2:
        return []

    continuations = []

    # Linear continuation
    diff = pattern[-1] - pattern[-2]
    next_linear = pattern[-1] + diff
    if min_num <= next_linear <= max_num:
        continuations.append(next_linear)

    # Geometric continuation
    if pattern[-2] != 0:
        ratio = pattern[-1] / pattern[-2]
        next_geometric = int(pattern[-1] * ratio)
        if min_num <= next_geometric <= max_num:
            continuations.append(next_geometric)

    # Fibonacci-like continuation
    if len(pattern) >= 3:
        next_fib = pattern[-1] + pattern[-2] - pattern[-3]  # Constant second difference
        if min_num <= next_fib <= max_num:
            continuations.append(next_fib)

    return continuations


def calculate_equilibrium_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate weights based on system equilibrium."""
    weights = defaultdict(float)

    # Calculate current system state
    if not past_draws:
        return weights

    current_draw = past_draws[0]
    if not current_draw:
        return weights

    # Calculate system metrics
    mean = statistics.mean(current_draw)
    std_dev = statistics.stdev(current_draw) if len(current_draw) > 1 else 0

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base equilibrium weight
        z_score = abs((num - mean) / std_dev) if std_dev > 0 else 0
        equilibrium_weight = np.exp(-z_score / 2)

        # Momentum adjustment
        momentum = calculate_system_momentum(past_draws)
        momentum_weight = 1 + momentum * (num - mean) / (max_num - min_num)

        # Energy distribution
        energy_weight = calculate_energy_distribution(num, past_draws)

        # Combine weights
        weights[num] = equilibrium_weight * momentum_weight * energy_weight

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def calculate_system_momentum(past_draws: List[List[int]]) -> float:
    """Calculate overall system momentum."""
    if len(past_draws) < 2:
        return 0

    momenta = []
    for i in range(len(past_draws) - 1):
        current_mean = statistics.mean(past_draws[i])
        next_mean = statistics.mean(past_draws[i + 1])
        momenta.append(next_mean - current_mean)

    return sum(momenta) / len(momenta) if momenta else 0


def calculate_energy_distribution(num: int, past_draws: List[List[int]]) -> float:
    """Calculate energy distribution weight for a number."""
    if not past_draws:
        return 1.0

    # Calculate energy levels
    energies = []
    for draw in past_draws:
        sorted_nums = sorted(draw)
        for i in range(len(sorted_nums) - 1):
            energies.append(sorted_nums[i + 1] - sorted_nums[i])

    if not energies:
        return 1.0

    # Calculate energy distribution
    mean_energy = statistics.mean(energies)
    std_energy = statistics.stdev(energies) if len(energies) > 1 else 1

    # Calculate number's energy compatibility
    last_draw = sorted(past_draws[0])
    nearest_energy = min(abs(num - n) for n in last_draw)

    # Score based on energy distribution
    energy_score = np.exp(-abs(nearest_energy - mean_energy) / std_energy)

    return energy_score


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers using equilibrium principles."""
    numbers = set()
    required_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Create balanced distribution
    mean = (lottery_type_instance.min_number + lottery_type_instance.max_number) / 2
    std = (lottery_type_instance.max_number - lottery_type_instance.min_number) / 6

    while len(numbers) < required_numbers:
        # Generate number from normal distribution
        num = int(random.gauss(mean, std))

        # Ensure number is within bounds
        num = max(lottery_type_instance.min_number,
                  min(lottery_type_instance.max_number, num))

        if num not in numbers:
            numbers.add(num)

    return sorted(list(numbers))