# modulo_pattern_prediction.py

from collections import Counter
from typing import List, Dict, Tuple, Set
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using modulo pattern analysis for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            is_main=False
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_numbers(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[int]:
    """
    Generate numbers using modulo pattern analysis.
    """
    if is_main:
        min_number = int(lottery_type_instance.min_number)
        max_number = int(lottery_type_instance.max_number)
        pieces_of_draw_numbers = int(lottery_type_instance.pieces_of_draw_numbers)
        numbers_field = 'lottery_type_number'
    else:
        min_number = int(lottery_type_instance.additional_min_number)
        max_number = int(lottery_type_instance.additional_max_number)
        pieces_of_draw_numbers = int(lottery_type_instance.additional_numbers_count)
        numbers_field = 'additional_numbers'

    # Fetch past draws
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(numbers_field, flat=True)

    past_draws = [draw for draw in past_draws_queryset if isinstance(draw, list)]

    modulo_patterns = analyze_modulo_patterns(past_draws, min_number, max_number)
    predicted_numbers = generate_numbers_from_patterns(modulo_patterns, min_number, max_number, pieces_of_draw_numbers)

    # Remove extra numbers if any
    while len(predicted_numbers) > pieces_of_draw_numbers:
        predicted_numbers.remove(random.choice(list(predicted_numbers)))

    # Fill missing numbers if not enough
    predicted_numbers = fill_missing_numbers(predicted_numbers, min_number, max_number, pieces_of_draw_numbers)

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    return sorted(predicted_numbers)[:pieces_of_draw_numbers]


def analyze_modulo_patterns(past_draws: List[List[int]], min_number: int, max_number: int) -> Dict[int, Counter]:
    patterns = {i: Counter() for i in range(2, 11)}  # Analyze patterns for modulo 2 to 10
    for draw in past_draws:
        for number in draw:
            for modulo in patterns:
                patterns[modulo][number % modulo] += 1
    return patterns


def generate_numbers_from_patterns(
        patterns: Dict[int, Counter],
        min_number: int,
        max_number: int,
        pieces_of_draw_numbers: int
) -> Set[int]:
    predicted_numbers = set()
    for modulo, counter in patterns.items():
        most_common = counter.most_common(2)
        for remainder, _ in most_common:
            candidates = [
                num for num in range(min_number, max_number + 1)
                if num % modulo == remainder
            ]
            if candidates:
                new_number = random.choice(candidates)
                predicted_numbers.add(new_number)
                if len(predicted_numbers) >= pieces_of_draw_numbers:
                    return predicted_numbers
    return predicted_numbers


def fill_missing_numbers(
        numbers: Set[int],
        min_number: int,
        max_number: int,
        pieces_of_draw_numbers: int
) -> List[int]:
    all_possible_numbers = set(range(min_number, max_number + 1))
    while len(numbers) < pieces_of_draw_numbers:
        remaining_numbers = all_possible_numbers - numbers
        if not remaining_numbers:
            break
        new_number = random.choice(list(remaining_numbers))
        numbers.add(new_number)
    return list(numbers)
