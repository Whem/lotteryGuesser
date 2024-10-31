# stable_evolution_manifold_prediction.py
# Combines stable pattern recognition with evolutionary manifold theory
# Uses stability analysis and dynamical systems approach

import numpy as np
import statistics
import random
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from scipy.spatial import distance
from sklearn.decomposition import PCA
from scipy import signal
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Stable Evolution Manifold Prediction.
    Combines stable pattern recognition with evolutionary manifold theory,
    leveraging stability analysis and dynamical systems approaches.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance=lottery_type_instance,
        number_field='lottery_type_number',
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count)
        )

    return main_numbers, additional_numbers


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using Stable Evolution Manifold Prediction.
    Combines stable pattern recognition with evolutionary manifold theory,
    leveraging stability analysis and dynamical systems approaches.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers
                    ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(number_field, flat=True).order_by('-id')[:200])

    # Filter valid past draws
    past_draws = [draw for draw in past_draws if isinstance(draw, list) and len(draw) == total_numbers]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance, min_num, max_num, total_numbers)

    candidates = set()

    # 1. Stable Manifold Analysis
    stable_numbers = analyze_stable_manifolds(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(stable_numbers[:max(total_numbers // 3, 1)])

    # 2. Evolution Surface Analysis
    evolution_numbers = analyze_evolution_surface(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(evolution_numbers[:max(total_numbers // 3, 1)])

    # 3. Stability Point Analysis
    stability_numbers = analyze_stability_points(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(stability_numbers[:max(total_numbers // 3, 1)])

    # Fill remaining slots using manifold probabilities
    while len(candidates) < total_numbers:
        weights = calculate_manifold_weights(
            past_draws,
            min_num,
            max_num,
            candidates
        )

        available_numbers = set(range(
            min_num,
            max_num + 1
        )) - candidates

        if not available_numbers:
            break  # No more numbers to select

        number_weights = [weights.get(num, 1.0) for num in available_numbers]
        selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
        candidates.add(selected)

    return sorted(list(candidates))[:total_numbers]


def analyze_stable_manifolds(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Analyze stable manifolds in the number space.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of predicted numbers based on stable manifold analysis.
    """
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
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            direction = vector / norm

            # Project direction to number space
            number_direction = pca.inverse_transform(direction)

            # Score numbers based on direction
            for num in range(min_num, max_num + 1):
                manifold_scores[num] += number_direction[num - min_num]

    except Exception as e:
        print(f"Manifold analysis error: {e}")

    return sorted(manifold_scores.keys(), key=manifold_scores.get, reverse=True)


def analyze_evolution_surface(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Analyze evolution surface patterns.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of predicted numbers based on evolution surface analysis.
    """
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
        fft_vals = np.abs(np.fft.fft(local_history))
        dominant_freq = np.argmax(fft_vals[1:]) + 1  # Skip the zero frequency
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
    """
    Analyze stability points in the system.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of predicted numbers based on stability point analysis.
    """
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
        self_transitions = transitions[num]
        total_transitions = sum(self_transitions.values())
        self_stability = transitions[num][num] / (total_transitions + 1) if total_transitions > 0 else 0

        # Neighborhood stability
        neighborhood = range(max(min_num, num - 5), min(max_num + 1, num + 6))
        neighborhood_stability = sum(transitions[num][n] for n in neighborhood) / (total_transitions + 1) if total_transitions > 0 else 0

        # Reverse stability (numbers that lead to this number)
        reverse_transitions = sum(transitions[n][num] for n in range(min_num, max_num + 1))
        total_reverse = sum(sum(transitions[n].values()) for n in range(min_num, max_num + 1))
        reverse_stability = reverse_transitions / (total_reverse + 1) if total_reverse > 0 else 0

        # Combine stability metrics
        stability_scores[num] = self_stability + neighborhood_stability + reverse_stability

    return sorted(stability_scores.keys(), key=stability_scores.get, reverse=True)


def calculate_manifold_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate weights based on manifold theory.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - excluded_numbers: Set of numbers to exclude from selection.

    Returns:
    - A dictionary mapping each number to its calculated weight.
    """
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
            min_distance = np.min(distances) if distances else 0

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
        # Fallback to uniform weights
        for num in range(min_num, max_num + 1):
            if num not in excluded_numbers:
                weights[num] = 1.0

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def generate_random_numbers(lottery_type_instance: lg_lottery_type, min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generate random numbers using manifold principles as a fallback mechanism.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of randomly generated lottery numbers.
    """
    numbers = set()
    required_numbers = total_numbers

    # Create manifold-guided distribution
    field = np.ones(max_num - min_num + 1)
    field = field / np.sum(field)  # Normalize

    while len(numbers) < required_numbers:
        # Sample from field distribution
        idx = np.random.choice(len(field), p=field)
        num = idx + min_num

        if num not in numbers:
            numbers.add(num)

            # Update field (reduce probability nearby)
            x = np.arange(len(field))
            width = len(field) / 20
            field *= (1 - 0.5 * np.exp(-(x - idx) ** 2 / (2 * width ** 2)))
            if np.sum(field) > 0:
                field = field / np.sum(field)  # Renormalize
            else:
                # Reset field if all probabilities are zero
                field = np.ones(len(field)) / len(field)

    return sorted(list(numbers))
