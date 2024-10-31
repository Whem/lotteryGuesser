# leapfrog_number_prediction.py

# Generates lottery numbers using the leapfrog pattern approach.
# Considers the last few draws and generates numbers by adding or subtracting a fixed leap to each number.
# Supports both main and additional numbers.

import random
from typing import List, Set, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using the leapfrog pattern for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_leapfrog_numbers(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_leapfrog_numbers(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_leapfrog_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    required_numbers: int,
    is_main: bool
) -> List[int]:
    """
    Generate numbers using the leapfrog pattern based on recent draws.
    """
    # Retrieve past draws
    if is_main:
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list('lottery_type_number', flat=True)
    else:
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list('additional_numbers', flat=True)

    past_draws = [draw for draw in past_draws_queryset if isinstance(draw, list)]
    predicted_numbers = set()

    if past_draws:
        last_draws = past_draws[:3]  # Consider last 3 draws
        for draw in last_draws:
            predicted_numbers.update(
                generate_numbers_from_draw(draw, min_num, max_num)
            )

    # Fill missing numbers if needed
    predicted_numbers = fill_missing_numbers(predicted_numbers, min_num, max_num, required_numbers)

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    return predicted_numbers[:required_numbers]

def generate_numbers_from_draw(
    draw: List[int],
    min_num: int,
    max_num: int
) -> Set[int]:
    """
    Generate leapfrog numbers from a single draw.
    """
    leapfrog_numbers = set()
    for number in draw:
        for leap in [-2, 2]:  # Consider both forward and backward leaps
            leapfrog_number = number + leap
            if min_num <= leapfrog_number <= max_num:
                leapfrog_numbers.add(leapfrog_number)
    return leapfrog_numbers

def fill_missing_numbers(
    numbers: Set[int],
    min_num: int,
    max_num: int,
    required_numbers: int
) -> List[int]:
    """
    Fill the set with random numbers until it reaches the required count.
    """
    all_possible_numbers = set(range(min_num, max_num + 1))
    while len(numbers) < required_numbers:
        remaining_numbers = all_possible_numbers - numbers
        if not remaining_numbers:
            break
        numbers.add(random.choice(list(remaining_numbers)))

    return sorted(numbers)
