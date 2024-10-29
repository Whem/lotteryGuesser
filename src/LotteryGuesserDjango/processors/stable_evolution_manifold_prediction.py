# stable_evolution_manifold_prediction.py
# Combines stable pattern recognition with evolutionary manifold theory
# Uses stability analysis and dynamical systems approach

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy.spatial import distance
from sklearn.decomposition import PCA
import random
from itertools import combinations


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Stable evolution algorithm that combines manifold learning
    with stable pattern recognition and dynamical systems theory.
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:200])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    candidates = set()

    # 1. Stable Manifold Analysis
    stable_numbers = analyze_stable_manifolds(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(stable_numbers[:required_numbers // 3])

    # 2. Evolution Surface Analysis
    evolution_numbers = analyze_evolution_surface(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(evolution_numbers[:required_numbers // 3])

    # 3. Stability Point Analysis
    stability_numbers = analyze_stability_points(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(stability_numbers[:required_numbers // 3])

    # Fill remaining slots using manifold probabilities
    while len(candidates) < required_numbers:
        weights = calculate_manifold_weights(
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


def analyze_stable_manifolds(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze stable manifolds in the number space."""
    manifold_scores = defaultdict(float)

    # Convert draws to high-dimensional space
    points = []
    for draw in past_draws:
        point = np.zeros(max_num - min_num + 1)
        for num in draw:
            point[num - min_num] = 1
        points.append(point)

    if not points:
        return []

    try:
        # Perform dimensionality reduction
        pca = PCA(n_components=min(len(points), 3))
        reduced_points = pca.fit_transform(points)

        # Find stable regions in reduced space
        for i in range(len(reduced_points)):
            # Calculate local stability
            neighbors = reduced_points[max(0, i - 5):min(len(reduced_points), i + 6)]
            if len(neighbors) < 2:
                continue

            center = np.mean(neighbors, axis=0)
            stability = 1 / (1 + np.mean([distance.euclidean(p, center) for p in neighbors]))

            # Project back to number space
            projection = pca.inverse_transform(center)

            # Score numbers based on stability
            for num in range(min_num, max_num + 1):
                manifold_scores[num] += stability * projection[num - min_num]

        # Analyze manifold connectivity
        for i in range(len(reduced_points) - 1):
            vector = reduced_points[i + 1] - reduced_points[i]
            direction = vector / np.linalg.norm(vector)

            # Project direction to number space
            number_direction = pca.inverse_transform(direction)

            # Score numbers based on direction
            for num in range(min_num, max_num + 1):
                manifold_scores[num] += number_direction[num - min_num]

    except Exception as e:
        print(f"Manifold analysis error: {e}")

    return sorted(manifold_scores.keys(), key=manifold_scores.get, reverse=True)


def analyze_evolution_surface(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze evolution surface patterns."""
    surface_scores = defaultdict(float)

    # Create evolution surface
    surface = np.zeros((max_num - min_num + 1, len(past_draws)))
    for j, draw in enumerate(past_draws):
        for num in draw:
            surface[num - min_num, j] = 1

    # Analyze surface patterns
    for i in range(surface.shape[0]):
        if np.sum(surface[i]) == 0:
            continue

        # Calculate local evolution metrics
        local_history = surface[i]

        # Frequency stability
        freq = np.mean(local_history)

        # Pattern stability
        changes = np.diff(local_history)
        stability = 1 / (1 + np.std(changes))

        # Periodic patterns
        fft = np.abs(np.fft.fft(local_history))
        dominant_freq = np.argmax(fft[1:]) + 1
        periodicity = len(local_history) / dominant_freq if dominant_freq > 0 else 0

        # Combine metrics
        surface_scores[i + min_num] = freq * stability * (1 + periodicity)

    # Analyze surface gradients
    gradients = np.gradient(surface, axis=1)
    for i in range(gradients.shape[0]):
        recent_gradient = np.mean(gradients[i, -5:])
        surface_scores[i + min_num] *= (1 + np.sign(recent_gradient) * np.abs(recent_gradient))

    return sorted(surface_scores.keys(), key=surface_scores.get, reverse=True)


def analyze_stability_points(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze stability points in the system."""
    stability_scores = defaultdict(float)

    # Create transition matrix
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(past_draws) - 1):
        current = set(past_draws[i])
        next_draw = set(past_draws[i + 1])

        for num1 in current:
            for num2 in next_draw:
                transitions[num1][num2] += 1

    # Find stable points
    for num in range(min_num, max_num + 1):
        # Self-transition stability
        self_stability = transitions[num][num] / (sum(transitions[num].values()) + 1)

        # Neighborhood stability
        neighborhood = range(max(min_num, num - 5), min(max_num + 1, num + 6))
        neighborhood_stability = sum(transitions[num][n] for n in neighborhood) / (sum(transitions[num].values()) + 1)

        # Reverse stability (numbers that lead to this number)
        reverse_transitions = sum(transitions[n][num] for n in range(min_num, max_num + 1))
        total_reverse = sum(sum(transitions[n].values()) for n in range(min_num, max_num + 1))
        reverse_stability = reverse_transitions / (total_reverse + 1)

        # Combine stability metrics
        stability_scores[num] = self_stability + neighborhood_stability + reverse_stability

    return sorted(stability_scores.keys(), key=stability_scores.get, reverse=True)


def calculate_manifold_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate weights based on manifold theory."""
    weights = defaultdict(float)

    if not past_draws:
        return weights

    try:
        # Create number vectors
        vectors = []
        for draw in past_draws:
            vector = np.zeros(max_num - min_num + 1)
            for num in draw:
                vector[num - min_num] = 1
            vectors.append(vector)

        # Calculate manifold metrics
        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Position in manifold
            idx = num - min_num
            position_vector = np.zeros_like(vectors[0])
            position_vector[idx] = 1

            # Distance to historical points
            distances = [distance.euclidean(position_vector, v) for v in vectors]
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)

            # Local density
            density = sum(1 / (1 + d) for d in distances)

            # Stability metric
            if len(distances) >= 2:
                stability = 1 / (1 + np.std(distances))
            else:
                stability = 1.0

            # Combine metrics
            weights[num] = (1 / (1 + avg_distance)) * (1 / (1 + min_distance)) * density * stability

    except Exception as e:
        print(f"Weight calculation error: {e}")
        return {num: 1.0 for num in range(min_num, max_num + 1) if num not in excluded_numbers}

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers using manifold principles."""
    numbers = set()
    required_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Create manifold-guided distribution
    center = (lottery_type_instance.min_number + lottery_type_instance.max_number) / 2
    spread = (lottery_type_instance.max_number - lottery_type_instance.min_number) / 4

    while len(numbers) < required_numbers:
        # Generate number from stable distribution
        angle = random.uniform(0, 2 * np.pi)
        radius = random.gauss(0, spread)

        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)

        num = int((x + y) / 2)
        num = max(lottery_type_instance.min_number,
                  min(lottery_type_instance.max_number, num))

        if num not in numbers:
            numbers.add(num)

    return sorted(list(numbers))