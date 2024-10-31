# topological_dynamic_system_predictor.py

import random
import math
from typing import List, Tuple, Set, Dict
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import datetime
import gudhi as gd
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def create_point_cloud(past_draws: List[List[int]], dimension: int) -> List[List[int]]:
    """
    Create a point cloud from past draws using time-delay embedding.

    Parameters:
    - past_draws: A list of past lottery number draws.
    - dimension: The embedding dimension for time-delay embedding.

    Returns:
    - A list representing the point cloud.
    """
    point_cloud = []
    for i in range(len(past_draws) - dimension + 1):
        point = [num for draw in past_draws[i:i + dimension] for num in draw]
        point_cloud.append(point)
    return point_cloud


def pairwise_distance(point1: List[int], point2: List[int]) -> float:
    """
    Calculate Euclidean distance between two points.

    Parameters:
    - point1: The first point as a list of integers.
    - point2: The second point as a list of integers.

    Returns:
    - The Euclidean distance as a float.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def calculate_persistent_homology(point_cloud: List[List[int]], max_distance: float) -> List[Tuple[int, int, float]]:
    """
    Simplified persistent homology calculation using a Rips complex.

    Parameters:
    - point_cloud: A list of points representing the point cloud.
    - max_distance: The maximum edge length to consider in the Rips complex.

    Returns:
    - A list of persistence tuples (dimension, pair, distance).
    """
    persistence = []
    try:
        # Create a Rips complex from the point cloud
        rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=max_distance)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()
    except Exception as e:
        print(f"Error in calculate_persistent_homology: {e}")
    return persistence


def lyapunov_exponent(sequence: List[int], embedding_dim: int, delay: int = 1) -> float:
    """
    Calculate Lyapunov exponent to measure chaoticity.

    Parameters:
    - sequence: A flattened list of past lottery numbers.
    - embedding_dim: The embedding dimension for the analysis.
    - delay: The delay parameter for time-delay embedding.

    Returns:
    - The Lyapunov exponent as a float.
    """
    N = len(sequence)
    M = N - (embedding_dim - 1) * delay

    y = sequence[:M]
    Y = [sequence[i:i + embedding_dim * delay:delay] for i in range(M)]

    epsilon = 1e-10
    lyap = 0
    count = 0
    for i in range(M):
        distances = [pairwise_distance(Y[i], Y[j]) for j in range(M) if i != j]
        if not distances:
            continue
        nearest = min(distances)
        if nearest < epsilon:
            continue

        j = distances.index(nearest)
        d_n = abs(y[i] - y[j])
        if d_n < epsilon:
            continue

        lyap += math.log(d_n / nearest)
        count += 1

    return lyap / count if count > 0 else 0.0


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Topological Dynamic System-based analysis.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    past lottery draws using Topological Data Analysis (TDA) and dynamic system measures like
    Lyapunov exponents. It identifies underlying topological features and chaoticity to predict
    future numbers, ensuring uniqueness and adherence to the specified range.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    try:
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

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in get_numbers: {str(e)}")
        # Fall back to random number generation
        min_num = int(lottery_type_instance.min_number)
        max_num = int(lottery_type_instance.max_number)
        total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)
        main_numbers = generate_random_numbers(min_num, max_num, total_numbers)

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_min_num = int(lottery_type_instance.additional_min_number)
            additional_max_num = int(lottery_type_instance.additional_max_number)
            additional_total_numbers = int(lottery_type_instance.additional_numbers_count)
            additional_numbers = generate_random_numbers(additional_min_num, additional_max_num, additional_total_numbers)

        return main_numbers, additional_numbers


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using Topological Dynamic System-based prediction.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to create a point cloud, computes
    persistent homology, calculates Lyapunov exponents, and generates predicted numbers
    based on topological features and chaoticity measures. Any remaining slots are filled
    with weighted random choices based on statistical analysis.

    Parameters:
    - lottery_type_instance: The lottery type instance.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers
                    ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Number of numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    try:
        # Retrieve past winning numbers
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id').values_list(number_field, flat=True)

        past_draws = [
            draw for draw in past_draws_queryset
            if isinstance(draw, list) and len(draw) == total_numbers
        ]

        if len(past_draws) < 10:
            # If not enough past draws, generate random numbers
            selected_numbers = generate_random_numbers(min_num, max_num, total_numbers)
            return selected_numbers

        # Create point cloud using time-delay embedding
        embedding_dim = 3
        point_cloud = create_point_cloud(past_draws, embedding_dim)

        # Calculate persistent homology
        max_distance = math.sqrt(total_numbers * (max_num - min_num) ** 2)
        persistence = calculate_persistent_homology(point_cloud, max_distance)

        # Extract topological features (e.g., persistence distances)
        topological_features = [p[2] for p in persistence if p[0] == 1]  # 1-dimensional features (loops)

        # Calculate Lyapunov exponent
        flat_sequence = [num for draw in past_draws for num in draw]
        lyap_exp = lyapunov_exponent(flat_sequence, embedding_dim, delay=1)

        # Generate numbers based on topological features and chaoticity
        predicted_numbers = []
        for _ in range(total_numbers * 2):
            if topological_features:
                feature = random.choice(topological_features)
                num = int(min_num + (feature / max_distance) * (max_num - min_num))
            else:
                num = random.randint(min_num, max_num)

            # Apply chaotic perturbation
            num = int(num + lyap_exp * (random.random() - 0.5) * (max_num - min_num))
            num = max(min_num, min(num, max_num))

            if num not in predicted_numbers:
                predicted_numbers.append(num)

        # Ensure uniqueness and correct range
        predicted_numbers = list(set(predicted_numbers))
        predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

        # If not enough unique numbers, fill with weighted random selection
        if len(predicted_numbers) < total_numbers:
            weights = calculate_field_weights(
                past_draws=past_draws,
                min_num=min_num,
                max_num=max_num,
                excluded_numbers=set(predicted_numbers)
            )

            while len(predicted_numbers) < total_numbers:
                number = weighted_random_choice(weights, set(range(min_num, max_num + 1)) - set(predicted_numbers))
                if number is not None:
                    predicted_numbers.append(number)

        # Sort and trim to required number of numbers
        selected_numbers = sorted(predicted_numbers)[:total_numbers]
        return selected_numbers

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in generate_numbers: {str(e)}")
        # Fallback to random number generation
        return generate_random_numbers(min_num, max_num, total_numbers)


def calculate_field_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate weights based on statistical similarity to past draws.

    This function assigns weights to each number based on how closely it aligns with the
    statistical properties (mean) of historical data. Numbers not yet selected are given higher
    weights if they are more statistically probable based on the average mean.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - excluded_numbers: Set of numbers to exclude from selection.

    Returns:
    - A dictionary mapping each number to its calculated weight.
    """
    from statistics import mean, stdev
    weights = {}

    if not past_draws:
        return weights

    try:
        # Calculate overall mean from past draws
        all_numbers = [num for draw in past_draws for num in draw]
        overall_mean = mean(all_numbers)
        overall_stdev = stdev(all_numbers) if len(all_numbers) > 1 else 1.0

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Calculate z-score for the number
            z_score = (num - overall_mean) / overall_stdev if overall_stdev != 0 else 0.0

            # Assign higher weight to numbers closer to the mean
            weight = max(0, 1 - abs(z_score))
            weights[num] = weight

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


def weighted_random_choice(weights: Dict[int, float], available_numbers: Set[int]) -> int:
    """
    Selects a random number based on weighted probabilities.

    Parameters:
    - weights: A dictionary mapping numbers to their weights.
    - available_numbers: A set of numbers available for selection.

    Returns:
    - A single selected number.
    """
    try:
        numbers = list(available_numbers)
        number_weights = [weights.get(num, 1.0) for num in numbers]
        total = sum(number_weights)
        if total == 0:
            return random.choice(numbers)
        probabilities = [w / total for w in number_weights]
        selected = random.choices(numbers, weights=probabilities, k=1)[0]
        return selected
    except Exception as e:
        print(f"Weighted random choice error: {e}")
        return random.choice(list(available_numbers)) if available_numbers else None


def generate_random_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generates a sorted list of unique random numbers within the specified range.

    Parameters:
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Number of numbers to generate.

    Returns:
    - A sorted list of randomly generated lottery numbers.
    """
    try:
        numbers = set()
        while len(numbers) < total_numbers:
            num = random.randint(min_num, max_num)
            numbers.add(num)
        return sorted(list(numbers))
    except Exception as e:
        print(f"Error in generate_random_numbers: {str(e)}")
        # As a last resort, return a sequential list
        return list(range(min_num, min_num + total_numbers))


# Optional: Additional helper functions can be added here if needed.
