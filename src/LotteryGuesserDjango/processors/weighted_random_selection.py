# weighted_random_selection.py

import random
from typing import List, Tuple, Set, Dict
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using weighted random selection based on past frequencies.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    historical lottery draws. It calculates the frequency of each number in past draws and uses these
    frequencies to assign weights for a weighted random selection. If there is insufficient historical
    data, it falls back to generating random numbers within the valid range.

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
    Generates a list of lottery numbers using weighted random selection based on past frequencies.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It performs the following steps:
    1. Retrieves historical lottery data.
    2. Counts the frequency of each number in past draws.
    3. Calculates weights based on frequencies.
    4. Performs weighted random selection without replacement.
    5. Ensures that the generated numbers are within the valid range and unique.
    6. Fills any remaining slots with the most common numbers if necessary.

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
            # If not enough past draws, generate random numbers
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Count frequency of each number in past draws
        number_counter = Counter()
        for draw in past_draws:
            for number in draw:
                if isinstance(number, int):
                    number_counter[number] += 1

        all_numbers = list(range(min_num, max_num + 1))

        # Calculate weights based on frequencies
        frequencies = [number_counter.get(number, 0) for number in all_numbers]

        # If all frequencies are zero (no past data), assign equal weights
        if all(f == 0 for f in frequencies):
            weights = [1.0] * len(all_numbers)
        else:
            # Normalize frequencies to get weights
            total_freq = sum(frequencies)
            weights = [freq / total_freq for freq in frequencies]

        # Select numbers based on weights without replacement
        selected_numbers = []
        available_numbers = all_numbers.copy()
        available_weights = weights.copy()

        for _ in range(total_numbers):
            if not available_numbers:
                break
            selected_number = random.choices(available_numbers, weights=available_weights, k=1)[0]
            selected_numbers.append(selected_number)
            # Remove selected number and its weight
            index = available_numbers.index(selected_number)
            del available_numbers[index]
            del available_weights[index]

        # Ensure uniqueness and correct count
        selected_numbers = list(set(selected_numbers))
        if len(selected_numbers) < total_numbers:
            # Fill the remaining slots with the most common numbers
            most_common = number_counter.most_common(total_numbers - len(selected_numbers))
            for num, _ in most_common:
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


def calculate_weights(
    past_draws: List[List[int]],
    min_num: int,
    max_num: int,
    excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate weights based on statistical similarity to past draws.

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
        # Flatten all numbers from past draws
        all_numbers = [num for draw in past_draws for num in draw]
        overall_mean = statistics.mean(all_numbers)
        overall_stdev = statistics.stdev(all_numbers) if len(all_numbers) > 1 else 1.0

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Calculate z-score
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
