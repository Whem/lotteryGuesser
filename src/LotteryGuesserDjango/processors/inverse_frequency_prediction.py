#inverse_frequency_prediction.py
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers based on inverse frequency analysis for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    required_numbers: int,
    is_main: bool
) -> List[int]:
    """
    Generate a set of numbers using inverse frequency analysis.
    """
    # Fetch past draws
    field_name = 'lottery_type_number' if is_main else 'additional_numbers'
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(field_name, flat=True)

    # Extract numbers based on whether they are main or additional
    past_draws = []
    for draw in past_draws_queryset:
        if isinstance(draw, list):
            if is_main:
                # Extract main numbers within the main number range
                numbers = [num for num in draw if min_num <= num <= max_num]
            else:
                # Extract additional numbers within the additional number range
                numbers = [num for num in draw if min_num <= num <= max_num]
            if numbers:
                past_draws.append(numbers)

    if not past_draws:
        # If no past draws, return random numbers
        return random.sample(range(min_num, max_num + 1), required_numbers)

    # Count frequency of each number
    frequency_counter = Counter(num for draw in past_draws for num in draw)

    total_draws = len(past_draws)
    inverse_frequency = {num: total_draws / (freq + 1) for num, freq in frequency_counter.items()}

    all_numbers = set(range(min_num, max_num + 1))
    never_drawn = all_numbers - set(frequency_counter.keys())

    for num in never_drawn:
        inverse_frequency[num] = total_draws

    # Sort numbers by inverse frequency
    predicted_numbers = sorted(inverse_frequency, key=inverse_frequency.get, reverse=True)

    # Select the required number of numbers
    predicted_numbers = predicted_numbers[:required_numbers]

    # Ensure all numbers are standard Python ints
    predicted_numbers = [int(num) for num in predicted_numbers]

    return predicted_numbers
