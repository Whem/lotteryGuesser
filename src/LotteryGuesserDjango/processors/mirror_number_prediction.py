# mirror_number_prediction.py

from typing import List, Set, Tuple
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using mirror number prediction for both main and additional numbers.
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
    Generate numbers using mirror number prediction.
    """
    # Fetch past draws
    if is_main:
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).values_list('lottery_type_number', flat=True)
    else:
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).values_list('additional_numbers', flat=True)

    past_draws = [draw for draw in past_draws_queryset if isinstance(draw, list)]

    mirror_numbers = generate_mirror_numbers(past_draws, min_num, max_num)

    predicted_numbers = set(mirror_numbers)
    predicted_numbers = fill_missing_numbers(predicted_numbers, min_num, max_num, required_numbers)

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    return sorted(predicted_numbers)[:required_numbers]

def generate_mirror_numbers(past_draws: List[List[int]], min_num: int, max_num: int) -> Set[int]:
    mirror_numbers = set()
    for draw in past_draws:
        for number in draw:
            mirror = get_mirror_number(number, min_num, max_num)
            if mirror:
                mirror_numbers.add(mirror)
    return mirror_numbers

def get_mirror_number(number: int, min_num: int, max_num: int) -> int:
    mirror = int(str(number)[::-1])
    if min_num <= mirror <= max_num:
        return mirror
    return 0

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
    return list(numbers)
