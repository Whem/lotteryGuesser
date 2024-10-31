# time_series_arima_prediction.py

import random
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import List, Tuple, Set, Dict
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import datetime
import warnings


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using ARIMA time series prediction.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    past lottery draws using the ARIMA model. It forecasts future numbers based on historical trends
    and ensures that the generated numbers are unique and within the specified range.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Suppress warnings from ARIMA
    warnings.filterwarnings("ignore")

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
    Generates a list of lottery numbers using ARIMA time series prediction.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to train an ARIMA model for each number position,
    predicts the next numbers, ensures uniqueness, and fills any remaining slots with weighted
    random choices based on statistical analysis.

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

        if len(past_draws) < 20:
            # Not enough data to train the model; fallback to random numbers
            selected_numbers = generate_random_numbers(min_num, max_num, total_numbers)
            return selected_numbers

        # Flatten the past numbers
        past_numbers = [num for draw in past_draws for num in draw if isinstance(num, int)]

        # Prepare the data for ARIMA
        df = pd.DataFrame({'numbers': past_numbers})

        # Fit the ARIMA model
        model = ARIMA(df['numbers'], order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast future numbers
        forecast_steps = total_numbers * 2  # Forecast more steps to ensure enough unique numbers
        forecast = model_fit.forecast(steps=forecast_steps)
        predicted_numbers = forecast.round().astype(int).tolist()

        # Filter numbers within the valid range and remove duplicates
        predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]
        predicted_numbers = list(dict.fromkeys(predicted_numbers))  # Preserve order and remove duplicates

        # Ensure we have enough numbers
        if len(predicted_numbers) < total_numbers:
            remaining_numbers = list(set(range(min_num, max_num + 1)) - set(predicted_numbers))
            random.shuffle(remaining_numbers)
            predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])

        else:
            predicted_numbers = predicted_numbers[:total_numbers]

        # Sort and return the numbers
        selected_numbers = sorted(predicted_numbers)
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
