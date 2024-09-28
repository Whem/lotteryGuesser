# geometric_progression_prediction.py

import random
from typing import List, Set  # Removed Tuple from imports
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Generates lottery numbers based on geometric progression patterns found in past draws.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past draws
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    past_draws = [draw for draw in past_draws_queryset if isinstance(draw, list)]

    # Find geometric progressions in past draws
    progressions = find_geometric_progressions(past_draws)

    # Generate numbers from the most common progressions
    predicted_numbers = generate_numbers_from_progressions(progressions, lottery_type_instance)

    # If we don't have enough numbers, fill with random ones
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        predicted_numbers.add(new_number)

    # Return the required number of sorted numbers
    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]


def find_geometric_progressions(past_draws: List[List[int]]) -> Counter:
    """
    Finds geometric progressions in past lottery draws.

    Parameters:
    - past_draws: A list of past draws, each draw is a list of numbers.

    Returns:
    - A Counter object with progressions as keys and their occurrence counts as values.
    """
    progressions = Counter()
    for draw in past_draws:
        sorted_draw = sorted(set(draw))  # Remove duplicates
        n = len(sorted_draw)
        # Only proceed if there are at least 3 unique numbers
        if n < 3:
            continue
        # Check for all combinations of 3 numbers in the draw
        for i in range(n - 2):
            a = sorted_draw[i]
            for j in range(i + 1, n - 1):
                b = sorted_draw[j]
                # Avoid division by zero
                if a == 0:
                    continue
                ratio = b / a
                # Ratio must be positive and not zero
                if ratio <= 0:
                    continue
                for k in range(j + 1, n):
                    c = sorted_draw[k]
                    if is_geometric_progression(a, b, c):
                        progression = (a, b, c)
                        progressions[progression] += 1
    return progressions


def is_geometric_progression(a: int, b: int, c: int) -> bool:
    """
    Checks if three numbers form a geometric progression.

    Parameters:
    - a, b, c: The three numbers to check.

    Returns:
    - True if they form a geometric progression, False otherwise.
    """
    # Avoid division by zero
    if a == 0 or b == 0:
        return False
    return b ** 2 == a * c


def generate_numbers_from_progressions(progressions: Counter, lottery_type_instance: lg_lottery_type) -> Set[int]:
    """
    Generates predicted numbers by extending the most common geometric progressions.

    Parameters:
    - progressions: A Counter object of geometric progressions.
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A set of predicted lottery numbers.
    """
    predicted_numbers = set()
    # Consider top N most common progressions
    top_progressions = progressions.most_common(3)

    for progression, _ in top_progressions:
        predicted_numbers.update(progression)

        # Extend the progression
        extend_progression(predicted_numbers, progression, lottery_type_instance)

        # Stop if we have enough numbers
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    return predicted_numbers


def extend_progression(predicted_numbers: Set[int], progression: tuple[int, int, int], lottery_type_instance: lg_lottery_type):
    """
    Extends a geometric progression forward and backward.

    Parameters:
    - predicted_numbers: A set to store predicted numbers.
    - progression: The geometric progression to extend.
    - lottery_type_instance: An instance of lg_lottery_type model.
    """
    a, b, c = progression
    # Calculate the common ratio
    ratio = b / a

    # Avoid invalid ratios
    if ratio <= 0 or ratio == 1:
        return

    # Extend forward
    next_number = c * ratio
    while (lottery_type_instance.min_number <= next_number <= lottery_type_instance.max_number
           and next_number.is_integer()):
        predicted_numbers.add(int(next_number))
        next_number *= ratio

    # Extend backward
    prev_number = a / ratio
    while (lottery_type_instance.min_number <= prev_number <= lottery_type_instance.max_number
           and prev_number.is_integer()):
        predicted_numbers.add(int(prev_number))
        prev_number /= ratio
