# standard_deviation_based_prediction.py

import random
from statistics import mean, stdev
from typing import List, Tuple, Set, Dict
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on standard deviation-based prediction.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    the standard deviations and means of past lottery draws. It ensures that the generated numbers
    have statistical properties similar to historical data.

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
    Generates a list of lottery numbers based on standard deviation-based prediction.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to calculate average means and standard
    deviations, then generates candidate numbers that statistically align with historical data.

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
    ).values_list(number_field, flat=True))

    # Calculate standard deviations and means of past draws
    stdevs = []
    means = []
    for draw in past_draws:
        if isinstance(draw, list) and len(draw) == total_numbers:
            nums = [num for num in draw if isinstance(num, int)]
            if len(nums) == total_numbers:
                draw_mean = mean(nums)
                draw_stdev = stdev(nums) if len(nums) > 1 else 0.0
                means.append(draw_mean)
                stdevs.append(draw_stdev)

    if stdevs and means:
        avg_stdev = mean(stdevs)
        avg_mean = mean(means)
    else:
        # If no past data, return random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Generate candidate sets matching average standard deviation and mean
    max_attempts = 1000
    stdev_tolerance = avg_stdev * 0.1  # 10% tolerance
    mean_tolerance = (max_num - min_num) * 0.05  # 5% of range

    for _ in range(max_attempts):
        candidate_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        candidate_mean = mean(candidate_numbers)
        candidate_stdev = stdev(candidate_numbers) if len(candidate_numbers) > 1 else 0.0
        if (abs(candidate_stdev - avg_stdev) <= stdev_tolerance and
            abs(candidate_mean - avg_mean) <= mean_tolerance):
            selected_numbers = sorted(candidate_numbers)
            return selected_numbers

    # If no suitable candidate found, return random numbers
    selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
    selected_numbers.sort()
    return selected_numbers


def calculate_field_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate weights based on statistical similarity to past draws.

    This function assigns weights to each number based on how closely it aligns with the
    statistical properties (mean and standard deviation) of historical data. Numbers
    not yet selected are given higher weights if they are more statistically probable.

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
        overall_mean = mean(all_numbers)
        overall_stdev = stdev(all_numbers) if len(all_numbers) > 1 else 1.0

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Calculate z-score for the number
            z_score = (num - overall_mean) / overall_stdev if overall_stdev != 0 else 0.0

            # Assign higher weight to numbers closer to the mean
            weight = max(0, 1 - abs(z_score))  # Normalize between 0 and 1
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
