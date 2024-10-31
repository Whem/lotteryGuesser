# xgboost_prediction.py

import random
import math
from typing import List, Tuple, Set, Dict
from collections import Counter, defaultdict
import numpy as np
import xgboost as xgb
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using XGBoost regression based on historical draw data.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    historical lottery draws. It trains an XGBoost regression model to predict the next set of numbers
    based on previous draws. If insufficient historical data is available, it falls back to generating
    random numbers within the valid range.

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
    Generates a list of lottery numbers using XGBoost regression based on historical frequencies.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It performs the following steps:
    1. Retrieves and preprocesses historical lottery data.
    2. Prepares the dataset for XGBoost regression.
    3. Trains an XGBoost model to predict the next draw.
    4. Generates predictions and ensures they are within the valid range and unique.
    5. Fills any remaining slots with the most common historical numbers if necessary.

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
        ).order_by('-id')[:200].values_list(number_field, flat=True)

        past_draws = [
            draw for draw in past_draws_queryset
            if isinstance(draw, list) and len(draw) == total_numbers
        ]

        if len(past_draws) < 10:
            # Not enough data to train the model
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Prepare training data
        X, y = prepare_training_data(past_draws)

        if X.size == 0 or y.size == 0:
            # If training data is empty, fall back to random numbers
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Train the XGBoost model
        model = train_xgboost_model(X, y)

        # Use the last draw as input for prediction
        last_draw = np.array(past_draws[-1]).reshape(1, -1)
        prediction = model.predict(last_draw)[0]

        # Extract predicted numbers
        predicted_numbers = [int(round(num)) for num in prediction]
        # Clamp numbers within the valid range
        predicted_numbers = [clamp_number(num, min_num, max_num) for num in predicted_numbers]
        # Ensure uniqueness
        predicted_numbers = list(set(predicted_numbers))

        # If we have enough predicted numbers
        if len(predicted_numbers) >= total_numbers:
            selected_numbers = predicted_numbers[:total_numbers]
        else:
            # Fill the remaining numbers with the most common historical numbers
            common_numbers = get_most_common_numbers(past_draws, total_numbers - len(predicted_numbers))
            selected_numbers = predicted_numbers + common_numbers

        # Ensure the correct number of numbers
        selected_numbers = selected_numbers[:total_numbers]
        selected_numbers.sort()
        return selected_numbers

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in generate_numbers: {str(e)}")
        # Fall back to random number generation
        return generate_random_numbers(min_num, max_num, total_numbers)


def prepare_training_data(past_draws: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares the training data for XGBoost regression.

    Converts past draws into input-output pairs where each input is a draw and the output is the subsequent draw.

    Parameters:
    - past_draws: A list of past lottery number draws.

    Returns:
    - A tuple containing:
        - X: Feature matrix for training.
        - y: Target matrix for training.
    """
    X = []
    y = []
    for i in range(len(past_draws) - 1):
        X.append(past_draws[i])
        y.append(past_draws[i + 1])

    X = np.array(X)
    y = np.array(y)
    return X, y


def train_xgboost_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    """
    Trains an XGBoost regression model on the provided dataset.

    Parameters:
    - X: Feature matrix for training.
    - y: Target matrix for training.

    Returns:
    - A trained XGBoost regressor model.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def clamp_number(num: int, min_num: int, max_num: int) -> int:
    """
    Clamps a number within the specified range.

    Parameters:
    - num: The number to clamp.
    - min_num: The minimum allowable number.
    - max_num: The maximum allowable number.

    Returns:
    - The clamped number.
    """
    return max(min_num, min(num, max_num))


def get_most_common_numbers(past_draws: List[List[int]], count: int) -> List[int]:
    """
    Retrieves the most common numbers from past draws.

    Parameters:
    - past_draws: A list of past lottery number draws.
    - count: The number of most common numbers to retrieve.

    Returns:
    - A list of the most common lottery numbers.
    """
    all_numbers = [number for draw in past_draws for number in draw]
    number_counts = Counter(all_numbers)
    most_common = number_counts.most_common(count)
    return [num for num, _ in most_common]


def generate_random_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generates a sorted list of unique random numbers within the specified range.

    This function serves as a fallback mechanism to ensure that a valid set of numbers is always returned,
    even when historical data is insufficient or prediction algorithms fail.

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
