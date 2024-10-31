# summation_sequence_prediction.py

import random
from statistics import mean
from typing import List, Tuple, Set, Dict
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on summation sequence prediction.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    the sequence of sums from past lottery draws. It predicts the next sum in the sequence and
    generates candidate numbers that closely match this predicted sum within a specified tolerance.

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
    Generates a list of lottery numbers based on summation sequence prediction.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to calculate average sums and predicts the
    next sum in the sequence based on the average difference between consecutive sums. It then
    generates candidate numbers that have a sum within a specified tolerance of the predicted sum.

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

    # Calculate sums of past draws
    sum_sequence = []
    for draw in past_draws:
        if isinstance(draw, list) and len(draw) == total_numbers:
            nums = [num for num in draw if isinstance(num, int)]
            if len(nums) == total_numbers:
                draw_sum = sum(nums)
                sum_sequence.append(draw_sum)

    if len(sum_sequence) >= 3:
        # Calculate differences between consecutive sums
        sum_differences = [sum_sequence[i+1] - sum_sequence[i] for i in range(len(sum_sequence)-1)]
        # Calculate average difference (assumes linear trend)
        avg_difference = mean(sum_differences)
        # Predict the next sum in the sequence
        predicted_next_sum = sum_sequence[-1] + avg_difference

        # Generate candidate sets matching the predicted sum
        max_attempts = 1000
        sum_tolerance = predicted_next_sum * 0.05  # 5% tolerance

        for _ in range(max_attempts):
            candidate_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
            candidate_sum = sum(candidate_numbers)
            if abs(candidate_sum - predicted_next_sum) <= sum_tolerance:
                selected_numbers = sorted(candidate_numbers)
                return selected_numbers

    # If not enough data or no suitable candidate found, use statistical weights
    weights = calculate_field_weights(
        past_draws=past_draws,
        min_num=min_num,
        max_num=max_num,
        excluded_numbers=set()
    )

    # Generate numbers based on calculated weights
    selected_numbers = []
    available_numbers = set(range(min_num, max_num + 1))
    while len(selected_numbers) < total_numbers:
        number = weighted_random_choice(weights, available_numbers)
        if number is not None:
            selected_numbers.append(number)
            available_numbers.remove(number)

    return sorted(selected_numbers)


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
        all_sums = [sum(draw) for draw in past_draws if isinstance(draw, list)]
        overall_mean = mean(all_sums)
        # For sum prediction, weights can be inversely related to distance from overall_mean
        target_num = overall_mean / len(past_draws[0]) if past_draws and past_draws[0] else 0.0

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Weight based on how much adding this number would approach the target_num
            weight = max(0, 1 - abs(num - target_num) / (max_num - min_num))
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
