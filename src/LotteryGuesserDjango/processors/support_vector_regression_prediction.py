# support_vector_regression_prediction.py

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Set, Dict
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random
from statistics import mean, stdev


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Support Vector Regression (SVR).

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    past lottery draws using Support Vector Regression. It predicts the next number in each position
    based on historical trends and ensures that the generated numbers are unique and within the specified range.

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
    Generates a list of lottery numbers using Support Vector Regression (SVR).

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to train SVR models for each number position,
    predicts the next number in each position, ensures uniqueness, and fills any remaining slots
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
    # Retrieve past winning numbers
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(number_field, flat=True).order_by('-id')[:200])

    # Filter valid past draws
    past_draws = [draw for draw in past_draws if isinstance(draw, list) and len(draw) == total_numbers]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance, min_num, max_num, total_numbers)

    # Convert past draws to a numpy matrix
    try:
        draw_matrix = np.array(past_draws)
    except Exception as e:
        print(f"Error converting past draws to matrix: {e}")
        return generate_random_numbers(lottery_type_instance, min_num, max_num, total_numbers)

    scaler = StandardScaler()
    predicted_numbers = []

    # Train an SVR model for each number position
    for i in range(total_numbers):
        try:
            # Prepare the input features (time index) and target variable (number at position i)
            X = np.arange(len(draw_matrix)).reshape(-1, 1)  # Time index as feature
            y = draw_matrix[:, i].astype(float)

            # Scale the target variable
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

            # Initialize and train the SVR model
            model = SVR(kernel='rbf')
            model.fit(X, y_scaled)

            # Predict the next value
            next_index = np.array([[len(draw_matrix)]])
            y_pred_scaled = model.predict(next_index)

            # Inverse transform to get the predicted number
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # Round to the nearest integer and clamp within the range
            predicted_number = int(round(y_pred[0]))
            predicted_number = max(min_num, min(max_num, predicted_number))

            predicted_numbers.append(predicted_number)
        except Exception as e:
            print(f"SVR prediction error for position {i}: {e}")
            continue

    # Ensure uniqueness
    predicted_numbers = list(set(predicted_numbers))

    # If fewer numbers than required, fill with weighted random choices
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

    # Sort the numbers and ensure the correct count
    predicted_numbers = sorted(predicted_numbers)[:total_numbers]
    return predicted_numbers


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
    weights = defaultdict(float)

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


def generate_random_numbers(lottery_type_instance: lg_lottery_type, min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generate random numbers as a fallback mechanism.

    This function generates random numbers while ensuring they fall within the specified
    range and are unique.

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

    while len(numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        numbers.add(num)

    return sorted(list(numbers))
