# fractal_network_evolution_prediction.py
# Combines fractal pattern analysis with dynamic network evolution
# and self-organizing criticality principles

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
import networkx as nx
from scipy.spatial.distance import euclidean
import random
import math
from itertools import combinations


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Fractal network evolution algorithm that combines self-similar patterns
    with network dynamics and critical point analysis.
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:250])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    candidates = set()

    # 1. Fractal Pattern Analysis
    fractal_numbers = analyze_fractal_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(fractal_numbers[:required_numbers // 3])

    # 2. Network Evolution Analysis
    network_numbers = analyze_network_evolution(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(network_numbers[:required_numbers // 3])

    # 3. Critical Point Analysis
    critical_numbers = analyze_critical_points(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(critical_numbers[:required_numbers // 3])

    # Fill remaining slots using dynamic weights
    while len(candidates) < required_numbers:
        weights = calculate_dynamic_weights(
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


def analyze_fractal_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze self-similar fractal patterns in the number sequence."""
    fractal_scores = defaultdict(float)

    # Convert draws to fractal dimension space
    for scale in [2, 3, 5, 8]:  # Different scaling factors
        for draw in past_draws:
            sorted_nums = sorted(draw)

            # Calculate fractal dimensions at different scales
            for i in range(len(sorted_nums) - scale + 1):
                subset = sorted_nums[i:i + scale]

                # Calculate self-similarity measure
                differences = [subset[j + 1] - subset[j] for j in range(len(subset) - 1)]
                if not differences:
                    continue

                # Box-counting dimension approximation
                dimension = calculate_box_dimension(differences)

                # Project next number based on fractal dimension
                if dimension > 0:
                    next_value = subset[-1] + (sum(differences) / len(differences)) * dimension
                    if min_num <= next_value <= max_num:
                        fractal_scores[int(next_value)] += 1 / scale

    # Analyze Mandelbrot-like sequences
    for draw in past_draws[:50]:  # Focus on recent draws
        sorted_nums = sorted(draw)
        for i in range(len(sorted_nums) - 2):
            z = complex(sorted_nums[i], sorted_nums[i + 1])
            c = complex(sorted_nums[i + 1], sorted_nums[i + 2])

            # Iterate Mandelbrot sequence
            for _ in range(5):
                z = z * z + c
                if min_num <= int(abs(z)) <= max_num:
                    fractal_scores[int(abs(z))] += 0.2

    return sorted(fractal_scores.keys(), key=fractal_scores.get, reverse=True)


def analyze_network_evolution(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze the evolution of number networks over time."""
    network_scores = defaultdict(float)

    # Create dynamic network
    G = nx.Graph()

    # Build network from historical data
    edge_weights = defaultdict(float)
    for i, draw in enumerate(past_draws):
        weight = 1 / (i + 1)  # Recent draws have higher weight

        # Add edges between numbers in same draw
        for num1, num2 in combinations(sorted(draw), 2):
            edge_weights[(num1, num2)] += weight

    # Create network with significant edges
    threshold = np.mean(list(edge_weights.values()))
    for (num1, num2), weight in edge_weights.items():
        if weight > threshold:
            G.add_edge(num1, num2, weight=weight)

    # Analyze network properties
    if G.number_of_nodes() > 0:
        # Centrality measures
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)

        # Score numbers based on network metrics
        for num in range(min_num, max_num + 1):
            if num in G:
                network_scores[num] += (
                        degree_cent.get(num, 0) +
                        betweenness_cent.get(num, 0) +
                        eigenvector_cent.get(num, 0)
                )

        # Analyze community structure
        communities = nx.community.greedy_modularity_communities(G)
        for i, community in enumerate(communities):
            # Predict numbers based on community structure
            comm_mean = sum(community) / len(community)
            comm_std = statistics.stdev(list(community)) if len(community) > 1 else 1

            for offset in [-2, -1, 1, 2]:
                predicted = int(comm_mean + offset * comm_std)
                if min_num <= predicted <= max_num:
                    network_scores[predicted] += 1 / (abs(offset) * (i + 1))

    return sorted(network_scores.keys(), key=network_scores.get, reverse=True)


def analyze_critical_points(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze critical points and phase transitions in the number sequence."""
    critical_scores = defaultdict(float)

    # Convert sequence to time series
    time_series = []
    for draw in past_draws:
        time_series.extend(sorted(draw))

    if not time_series:
        return []

    # Calculate fluctuations at different scales
    for window in [3, 5, 8, 13]:
        if len(time_series) < window:
            continue

        # Calculate local fluctuations
        fluctuations = []
        for i in range(len(time_series) - window):
            segment = time_series[i:i + window]
            fluct = np.std(segment)
            fluctuations.append(fluct)

        if not fluctuations:
            continue

        # Find critical points (large fluctuations)
        mean_fluct = np.mean(fluctuations)
        std_fluct = np.std(fluctuations)

        for i, fluct in enumerate(fluctuations):
            if fluct > mean_fluct + std_fluct:
                # Critical point found, analyze surrounding numbers
                critical_segment = time_series[i:i + window]

                # Project numbers based on critical behavior
                for num in range(min_num, max_num + 1):
                    # Distance from critical point
                    distances = [abs(num - x) for x in critical_segment]
                    min_distance = min(distances)

                    # Score inversely proportional to distance
                    critical_scores[num] += 1 / (1 + min_distance * window)

    # Analyze phase transitions
    phase_changes = []
    for i in range(len(time_series) - 1):
        diff = abs(time_series[i + 1] - time_series[i])
        if diff > np.mean(time_series):
            phase_changes.append(i)

    # Score numbers near phase transitions
    for phase_idx in phase_changes:
        if phase_idx < len(time_series):
            value = time_series[phase_idx]
            for offset in [-3, -2, -1, 1, 2, 3]:
                predicted = value + offset
                if min_num <= predicted <= max_num:
                    critical_scores[predicted] += 1 / (abs(offset) + 1)

    return sorted(critical_scores.keys(), key=critical_scores.get, reverse=True)


def calculate_box_dimension(sequence: List[float]) -> float:
    """Calculate box-counting dimension approximation."""
    if not sequence:
        return 0

    # Create bins for box counting
    bins = int(np.sqrt(len(sequence)))
    if bins < 2:
        return 0

    try:
        hist, _ = np.histogram(sequence, bins=bins)
        non_empty_boxes = np.count_nonzero(hist)
        if non_empty_boxes <= 1:
            return 0

        return np.log(non_empty_boxes) / np.log(bins)
    except:
        return 0


def calculate_dynamic_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate dynamic weights using multiple metrics."""
    weights = defaultdict(float)

    # Convert past draws to frequency distribution
    freq_dist = Counter(num for draw in past_draws for num in draw)
    total_numbers = sum(freq_dist.values())

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base frequency weight
        freq_weight = freq_dist.get(num, 0) / total_numbers

        # Recency weight
        recency = 0
        for i, draw in enumerate(past_draws):
            if num in draw:
                recency = 1 / (i + 1)
                break

        # Pattern weight
        pattern_weight = 0
        for i in range(len(past_draws) - 1):
            if num in past_draws[i] and num in past_draws[i + 1]:
                pattern_weight += 1 / (i + 1)

        # Distance from mean
        all_numbers = [n for draw in past_draws for n in draw]
        if all_numbers:
            mean = statistics.mean(all_numbers)
            std_dev = statistics.stdev(all_numbers)
            z_score = abs((num - mean) / std_dev)
            distance_weight = math.exp(-z_score / 2)
        else:
            distance_weight = 1.0

        # Combine weights
        weights[num] = (freq_weight + recency + pattern_weight + distance_weight) / 4

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

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