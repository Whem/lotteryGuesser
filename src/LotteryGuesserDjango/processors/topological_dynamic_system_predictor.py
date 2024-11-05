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


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Topological Dynamic System-based analysis.
    Returns a single list containing both main and additional numbers (if applicable).

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list combining main numbers and additional numbers (if applicable).
    """
    try:
        # Generate main numbers
        main_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='lottery_type_number',
            min_num=lottery_type_instance.min_number,
            max_num=lottery_type_instance.max_number,
            total_numbers=lottery_type_instance.pieces_of_draw_numbers
        )

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            # Generate additional numbers
            additional_numbers = generate_numbers(
                lottery_type_instance=lottery_type_instance,
                number_field='additional_numbers',
                min_num=lottery_type_instance.additional_min_number,
                max_num=lottery_type_instance.additional_max_number,
                total_numbers=lottery_type_instance.additional_numbers_count
            )

        # Return combined list
        return main_numbers,additional_numbers

    except Exception as e:
        print(f"Error in get_numbers: {str(e)}")
        # Fallback to random numbers
        main_numbers = generate_random_numbers(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number,
            lottery_type_instance.pieces_of_draw_numbers
        )

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_random_numbers(
                lottery_type_instance.additional_min_number,
                lottery_type_instance.additional_max_number,
                lottery_type_instance.additional_numbers_count
            )

        return main_numbers , additional_numbers


def generate_numbers(
        lottery_type_instance: lg_lottery_type,
        number_field: str,
        min_num: int,
        max_num: int,
        total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using Topological Dynamic System-based prediction.
    """
    try:
        # Retrieve past winning numbers
        past_draws = []
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id')

        for draw in past_draws_queryset:
            numbers = getattr(draw, number_field, None)
            if isinstance(numbers, list) and len(numbers) == total_numbers:
                past_draws.append(numbers)

        if len(past_draws) < 10:
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Create point cloud
        embedding_dim = 3
        point_cloud = create_point_cloud(past_draws, embedding_dim)

        if not point_cloud:
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Calculate persistent homology with safety checks
        max_distance = math.sqrt(total_numbers * (max_num - min_num) ** 2)
        persistence = calculate_persistent_homology(point_cloud, max_distance)

        # Extract topological features safely
        topological_features = []
        if persistence:
            topological_features = [p[2] for p in persistence if len(p) > 2 and p[0] == 1]

        # Calculate Lyapunov exponent
        flat_sequence = [num for draw in past_draws for num in draw]
        lyap_exp = lyapunov_exponent(flat_sequence, embedding_dim, delay=1)

        # Generate initial predictions
        predicted_numbers = set()
        attempts = 0
        max_attempts = total_numbers * 4

        while len(predicted_numbers) < total_numbers and attempts < max_attempts:
            attempts += 1

            if topological_features:
                feature = random.choice(topological_features)
                num = int(min_num + (feature / max_distance) * (max_num - min_num))
            else:
                num = random.randint(min_num, max_num)

            # Apply chaotic perturbation
            perturbation = lyap_exp * (random.random() - 0.5) * (max_num - min_num)
            num = int(num + perturbation)
            num = max(min_num, min(num, max_num))

            if min_num <= num <= max_num:
                predicted_numbers.add(num)

        # Fill remaining numbers using weighted selection
        if len(predicted_numbers) < total_numbers:
            weights = calculate_field_weights(
                past_draws=past_draws,
                min_num=min_num,
                max_num=max_num,
                excluded_numbers=predicted_numbers
            )

            remaining_numbers = set(range(min_num, max_num + 1)) - predicted_numbers
            while len(predicted_numbers) < total_numbers and remaining_numbers:
                selected = weighted_random_choice(weights, remaining_numbers)
                if selected is not None:
                    predicted_numbers.add(selected)
                    remaining_numbers.remove(selected)

        # Ensure we have exactly the required number of numbers
        if len(predicted_numbers) > total_numbers:
            predicted_numbers = set(sorted(predicted_numbers)[:total_numbers])
        elif len(predicted_numbers) < total_numbers:
            remaining = total_numbers - len(predicted_numbers)
            available = set(range(min_num, max_num + 1)) - predicted_numbers
            if available:
                predicted_numbers.update(random.sample(list(available), min(remaining, len(available))))

        return sorted(list(predicted_numbers))

    except Exception as e:
        print(f"Error in generate_numbers: {str(e)}")
        return generate_random_numbers(min_num, max_num, total_numbers)


def create_point_cloud(past_draws: List[List[int]], dimension: int) -> List[List[int]]:
    """Create a point cloud from past draws using time-delay embedding."""
    try:
        if not past_draws or dimension <= 0:
            return []

        point_cloud = []
        for i in range(len(past_draws) - dimension + 1):
            point = []
            for j in range(dimension):
                if i + j < len(past_draws):
                    point.extend(past_draws[i + j])
            if point:
                point_cloud.append(point)
        return point_cloud
    except Exception as e:
        print(f"Error in create_point_cloud: {str(e)}")
        return []


def calculate_persistent_homology(point_cloud: List[List[int]], max_distance: float) -> List[Tuple]:
    """Calculate persistent homology using a Rips complex."""
    try:
        if not point_cloud:
            return []

        rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=max_distance)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        return simplex_tree.persistence()
    except Exception as e:
        print(f"Error in calculate_persistent_homology: {str(e)}")
        return []


def lyapunov_exponent(sequence: List[int], embedding_dim: int, delay: int = 1) -> float:
    """Calculate Lyapunov exponent with safety checks."""
    try:
        if not sequence or embedding_dim <= 0:
            return 0.0

        N = len(sequence)
        if N < embedding_dim:
            return 0.0

        M = N - (embedding_dim - 1) * delay
        if M <= 0:
            return 0.0

        y = sequence[:M]
        Y = [sequence[i:i + embedding_dim * delay:delay] for i in range(M)]

        epsilon = 1e-10
        lyap = 0.0
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

    except Exception as e:
        print(f"Error in lyapunov_exponent: {str(e)}")
        return 0.0


def pairwise_distance(point1: List[int], point2: List[int]) -> float:
    """Calculate Euclidean distance between two points safely."""
    try:
        if len(point1) != len(point2):
            return float('inf')
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    except Exception:
        return float('inf')


def calculate_field_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate weights for number selection."""
    weights = {}
    try:
        if not past_draws:
            return {num: 1.0 for num in range(min_num, max_num + 1)
                    if num not in excluded_numbers}

        # Calculate statistics
        all_numbers = [num for draw in past_draws for num in draw]
        if not all_numbers:
            return {num: 1.0 for num in range(min_num, max_num + 1)
                    if num not in excluded_numbers}

        mean_val = sum(all_numbers) / len(all_numbers)
        variance = sum((x - mean_val) ** 2 for x in all_numbers) / len(all_numbers)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        # Calculate weights
        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue
            z_score = abs(num - mean_val) / std_dev
            weight = math.exp(-z_score / 2)  # Use exponential decay
            weights[num] = weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

    except Exception as e:
        print(f"Error in calculate_field_weights: {str(e)}")
        # Fallback to uniform weights
        weights = {num: 1.0 for num in range(min_num, max_num + 1)
                   if num not in excluded_numbers}

    return weights


def weighted_random_choice(weights: Dict[int, float], available_numbers: Set[int]) -> int:
    """Select a random number based on weights."""
    try:
        if not available_numbers:
            return None

        numbers = list(available_numbers)
        if not numbers:
            return None

        weights_list = [weights.get(num, 0.0) for num in numbers]
        total_weight = sum(weights_list)

        if total_weight <= 0:
            return random.choice(numbers)

        weights_list = [w / total_weight for w in weights_list]
        return random.choices(numbers, weights=weights_list, k=1)[0]

    except Exception as e:
        print(f"Error in weighted_random_choice: {str(e)}")
        return random.choice(list(available_numbers)) if available_numbers else None


def generate_random_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """Generate random numbers safely."""
    try:
        if max_num < min_num or total_numbers <= 0:
            return []

        numbers = set()
        available_range = list(range(min_num, max_num + 1))

        if total_numbers > len(available_range):
            total_numbers = len(available_range)

        while len(numbers) < total_numbers:
            numbers.add(random.choice(available_range))

        return sorted(list(numbers))
    except Exception as e:
        print(f"Error in generate_random_numbers: {str(e)}")
        return list(range(min_num, min(min_num + total_numbers, max_num + 1)))