#arithmetic_progression_prediction.py
import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate main and additional numbers based on arithmetic progression patterns.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    progression_differences = Counter()

    # Calculate differences from past draws to determine common progression patterns
    for draw in past_draws:
        sorted_draw = sorted(draw) if isinstance(draw, list) else []
        for i in range(len(sorted_draw) - 1):
            difference = sorted_draw[i + 1] - sorted_draw[i]
            progression_differences[difference] += 1

    # Determine the most common difference for the main numbers
    if not progression_differences:
        most_common_difference = random.randint(1, (
                    lottery_type_instance.max_number - lottery_type_instance.min_number) // lottery_type_instance.pieces_of_draw_numbers)
    else:
        most_common_difference = progression_differences.most_common(1)[0][0]

    main_numbers = generate_progression_set(
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        most_common_difference
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_most_common_difference = random.randint(1, (
                    lottery_type_instance.additional_max_number - lottery_type_instance.additional_min_number) // lottery_type_instance.additional_numbers_count)
        additional_numbers = generate_progression_set(
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            additional_most_common_difference
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_progression_set(min_num: int, max_num: int, required_numbers: int, difference: int) -> List[int]:
    """Generate a set of numbers based on a starting number and a progression difference."""
    valid_numbers = []
    attempts = 0
    max_attempts = 100  # Prevent infinite loop

    while len(valid_numbers) < required_numbers and attempts < max_attempts:
        start_number = random.randint(min_num, max_num)
        predicted_numbers = [start_number + i * difference for i in range(required_numbers)]

        # Filter numbers that fall within the valid range
        valid_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]
        attempts += 1

    # Fall back to random selection if a valid sequence could not be generated
    if len(valid_numbers) < required_numbers:
        valid_numbers = random.sample(range(min_num, max_num + 1), required_numbers)

    return valid_numbers
