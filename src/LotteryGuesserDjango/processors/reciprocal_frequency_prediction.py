# reciprocal_frequency_prediction.py
import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on repeating pattern prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance=lottery_type_instance,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers,
        number_field='lottery_type_number',
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance=lottery_type_instance,
            total_numbers=lottery_type_instance.additional_numbers_count,
            number_field='additional_numbers',
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number
        )

    return main_numbers, additional_numbers


def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    total_numbers: int,
    number_field: str,
    min_num: int,
    max_num: int
) -> List[int]:
    """
    Generates a set of lottery numbers based on repeating pattern prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - total_numbers: Total numbers to generate.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(number_field, flat=True)

    past_draws = [
        tuple(sorted(draw)) for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    # Count frequency of each pattern
    pattern_counter = Counter(past_draws)

    # Find repeating patterns
    repeating_patterns = [pattern for pattern, count in pattern_counter.items() if count > 1]

    if repeating_patterns:
        # Choose the most frequent repeating pattern
        most_common_pattern = max(repeating_patterns, key=lambda p: pattern_counter[p])
        selected_numbers = list(most_common_pattern)
    else:
        # If no repeating patterns, select random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)

    # Ensure unique numbers and correct count
    selected_numbers = list(set(selected_numbers))
    total_needed = total_numbers
    if len(selected_numbers) < total_needed:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_needed - len(selected_numbers)])
    elif len(selected_numbers) > total_needed:
        selected_numbers = selected_numbers[:total_needed]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
