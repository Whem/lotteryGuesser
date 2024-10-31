# symbolic_regression_prediction.py

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from typing import List, Tuple, Set, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random
from collections import defaultdict

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Symbolic Regression (SR) analysis.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    past lottery draws using Symbolic Regression. It predicts the next set of numbers based on
    learned symbolic expressions that model the relationship between historical draws.

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
    Generates a list of lottery numbers using Symbolic Regression (SR).

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to train SR models for each number position,
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
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(number_field, flat=True)

    past_draws = [
        [int(num) for num in draw] for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 20:
        # If not enough past draws, generate a default set of numbers
        selected_numbers = generate_default_numbers(min_num, max_num, total_numbers)
        return selected_numbers

    # Prepare features and targets for Symbolic Regression
    window_size = 5  # Number of past draws to consider for prediction
    X, y = create_features_and_targets(past_draws, window_size, total_numbers)

    if X.size == 0 or y.size == 0:
        # If feature or target sets are empty, fallback to random numbers
        selected_numbers = generate_default_numbers(min_num, max_num, total_numbers)
        return selected_numbers

    predicted_numbers = []

    # Train and predict for each number position
    for pos in range(total_numbers):
        try:
            # Define a custom fitness function (Mean Squared Error)
            mse = make_fitness(function=lambda y, y_pred, _: np.mean((y - y_pred) ** 2),
                               greater_is_better=False)

            # Initialize Symbolic Regressor
            est = SymbolicRegressor(
                population_size=1000,
                generations=20,
                tournament_size=20,
                stopping_criteria=0.01,
                const_range=(0, 10),
                init_depth=(2, 6),
                function_set=['add', 'sub', 'mul', 'div'],
                metric=mse,
                parsimony_coefficient=0.001,
                max_samples=1.0,
                verbose=0,
                n_jobs=1,
                random_state=0
            )

            # Fit the model
            est.fit(X, y[:, pos])

            # Predict the next number using the last window
            last_window = past_draws[-window_size:]
            flattened_last_window = np.array([num for draw in last_window for num in draw]).reshape(1, -1)
            predicted_value = est.predict(flattened_last_window)[0]

            # Round and clamp the predicted number within the valid range
            predicted_number = int(round(predicted_value))
            predicted_number = max(min_num, min(max_num, predicted_number))

            predicted_numbers.append(predicted_number)

        except Exception as e:
            print(f"Symbolic Regression prediction error for position {pos}: {e}")
            continue

    # Ensure uniqueness
    predicted_numbers = list(dict.fromkeys(predicted_numbers))

    # If not enough unique numbers, fill with weighted random choices
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

    # Sort and trim to the required number of numbers
    predicted_numbers = sorted(predicted_numbers)[:total_numbers]
    return predicted_numbers


def create_features_and_targets(past_draws: List[List[int]], window_size: int, total_numbers: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates feature and target datasets for Symbolic Regression.

    Parameters:
    - past_draws: List of past lottery number draws.
    - window_size: Number of past draws to consider for each feature.
    - total_numbers: Number of numbers in each draw.

    Returns:
    - A tuple containing:
        - X: Feature matrix.
        - y: Target matrix.
    """
    X = []
    y = []

    for i in range(len(past_draws) - window_size):
        window = past_draws[i:i + window_size]
        next_draw = past_draws[i + window_size]
        # Flatten the window into a single feature vector
        flattened_window = [num for draw in window for num in draw]
        X.append(flattened_window)
        y.append(next_draw)

    X = np.array(X)
    y = np.array(y)

    return X, y


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


def generate_default_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generates a default set of lottery numbers when insufficient historical data is available.

    Parameters:
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Number of numbers to generate.

    Returns:
    - A sorted list of default lottery numbers.
    """
    selected_numbers = list(range(min_num, min_num + total_numbers))
    return selected_numbers


def generate_random_numbers(lottery_type_instance: lg_lottery_type, min_num: int, max_num: int,
                           total_numbers: int) -> List[int]:
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
