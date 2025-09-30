# last_digit_pattern_prediction.py

from collections import Counter
from typing import List, Set, Tuple
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers based on last digit patterns for both main and additional numbers.
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
    Generate numbers based on last digit patterns.
    """
    # Fetch past draws from the correct field
    field_name = 'lottery_type_number' if is_main else 'additional_numbers'
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(field_name, flat=True)

    # Extract numbers
    past_numbers = []
    for draw in past_draws_queryset:
        if isinstance(draw, list):
            numbers = [num for num in draw if min_num <= num <= max_num]
            if numbers:
                past_numbers.extend(numbers)

    if not past_numbers:
        # If no past numbers, return random numbers
        return random.sample(range(min_num, max_num + 1), required_numbers)

    # Calculate last digit frequencies
    last_digit_counter = Counter(number % 10 for number in past_numbers)

    # Get most common last digits
    most_common_last_digits = [digit for digit, _ in last_digit_counter.most_common()]

    # Generate numbers from most common last digits
    predicted_numbers = generate_numbers_from_digits(
        most_common_last_digits,
        min_num,
        max_num,
        required_numbers
    )

    return predicted_numbers

def generate_numbers_from_digits(
    digits: List[int],
    min_num: int,
    max_num: int,
    required_numbers: int
) -> List[int]:
    numbers = set()
    for digit in digits:
        candidates = [
            num for num in range(min_num, max_num + 1)
            if num % 10 == digit
        ]
        if candidates:
            chosen_num = random.choice(candidates)
            numbers.add(chosen_num)
        if len(numbers) >= required_numbers:
            break

    # Fill missing numbers if needed
    all_possible_numbers = set(range(min_num, max_num + 1))
    while len(numbers) < required_numbers:
        remaining_numbers = all_possible_numbers - numbers
        if not remaining_numbers:
            break
        numbers.add(random.choice(list(remaining_numbers)))

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in numbers]

    return predicted_numbers[:required_numbers]
