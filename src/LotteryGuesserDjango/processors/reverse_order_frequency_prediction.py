# reverse_order_frequency_prediction.py
import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on reverse order frequency prediction.

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

    return main_numbers, additional_numbers

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers based on reverse order frequency prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(number_field, flat=True)

    # Count frequency of each number
    number_counter = Counter()
    for draw in past_draws:
        if not isinstance(draw, list):
            continue
        for number in draw:
            if isinstance(number, int):
                number_counter[number] += 1

    all_numbers = list(range(min_num, max_num + 1))

    # Sort numbers by increasing frequency (least frequent first)
    numbers_by_frequency = sorted(all_numbers, key=lambda x: number_counter.get(x, 0))

    # Select the least frequent numbers
    selected_numbers = numbers_by_frequency[:total_numbers]

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < total_numbers:
        remaining_numbers = list(set(all_numbers) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Ensure we have the correct number of numbers
    selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
