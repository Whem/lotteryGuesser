# wavelet_transform_prediction.py

import random
import math
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import numpy as np
import pywt  # Ensure that the PyWavelets library is installed
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on historical draw wavelet transform analysis.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    historical lottery draws using wavelet transform techniques. It applies discrete wavelet
    transforms to identify trends and patterns, predicts the next values based on these patterns,
    and ensures the generated numbers are within the specified range and unique.

    Parameters:
    - lottery_type_instance: An instance of the lg_lottery_type model.

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
            additional_numbers = generate_random_numbers(additional_min_num, additional_max_num,
                                                         additional_total_numbers)

        return main_numbers, additional_numbers


def generate_numbers(
        lottery_type_instance: lg_lottery_type,
        number_field: str,
        min_num: int,
        max_num: int,
        total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using wavelet transform-based prediction.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It performs the following steps:
    1. Retrieves and preprocesses historical lottery data.
    2. Applies discrete wavelet transforms to each number position to identify trends.
    3. Predicts the next number in each position based on the reconstructed signal.
    4. Ensures that the generated numbers are within the valid range and unique.
    5. Fills any remaining slots with the most common numbers from historical data if needed.

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
        ).order_by('id')[:200].values_list(number_field, flat=True)

        past_draws = [
            draw for draw in past_draws_queryset
            if isinstance(draw, list) and len(draw) == total_numbers
        ]

        if len(past_draws) < 10:
            # If not enough past draws, generate random numbers
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Convert past draws to a NumPy array for processing
        draw_matrix = np.array(past_draws)

        # Generate predictions for each number position
        predicted_numbers = []
        for i in range(total_numbers):
            series = draw_matrix[:, i]

            # Apply discrete wavelet transform
            coeffs = pywt.wavedec(series, 'db1', level=2)

            # Zero out detail coefficients to focus on the trend (approximation)
            coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]

            # Reconstruct the signal from approximation coefficients
            reconstructed_series = pywt.waverec(coeffs, 'db1')

            # Predict the next value based on the trend
            if len(reconstructed_series) >= 2:
                trend = reconstructed_series[-1] - reconstructed_series[-2]
                next_value = reconstructed_series[-1] + trend
            else:
                next_value = reconstructed_series[-1]

            # Round and clamp the predicted number within the valid range
            predicted_number = int(round(next_value))
            predicted_number = max(min_num, min(predicted_number, max_num))

            predicted_numbers.append(predicted_number)

        # Ensure uniqueness
        selected_numbers = list(set(predicted_numbers))

        # If not enough unique numbers, fill with the most common historical numbers
        if len(selected_numbers) < total_numbers:
            all_numbers = [num for draw in past_draws for num in draw]
            number_counts = Counter(all_numbers)
            sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
            for num, _ in sorted_numbers:
                if num not in selected_numbers:
                    selected_numbers.append(num)
                if len(selected_numbers) == total_numbers:
                    break

        # Trim to the required number of numbers and sort
        selected_numbers = selected_numbers[:total_numbers]
        selected_numbers.sort()
        return selected_numbers

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in generate_numbers: {str(e)}")
        # Fall back to random number generation
        return generate_random_numbers(min_num, max_num, total_numbers)


def calculate_wave_probabilities(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate probabilities based on wave characteristics for weighted random selection.

    This function assigns weights to each number based on how closely it aligns with the
    statistical properties (mean and standard deviation) of historical data. Numbers not
    yet selected are given higher weights if they are more statistically probable.

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
        # Calculate overall mean and standard deviation from past draws
        all_numbers = [num for draw in past_draws for num in draw]
        overall_mean = statistics.mean(all_numbers)
        overall_stdev = statistics.stdev(all_numbers) if len(all_numbers) > 1 else 1.0

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Calculate z-score for the number
            z_score = abs((num - overall_mean) / overall_stdev) if overall_stdev != 0 else 0.0

            # Assign higher weight to numbers closer to the mean
            weight = max(0, 1 - z_score)
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
