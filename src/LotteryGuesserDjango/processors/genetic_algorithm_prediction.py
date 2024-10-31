# genetic_algorithm_prediction.py
import random
from collections import Counter
from typing import List, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Generates numbers for both main and additional sets using geometric progression."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generates a set of numbers using geometric progression analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)
    progressions = find_geometric_progressions(past_draws)
    predicted_numbers = generate_numbers_from_progressions(
        progressions, min_num, max_num, required_numbers
    )

    while len(predicted_numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        predicted_numbers.add(new_number)

    return sorted(list(predicted_numbers))[:required_numbers]


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Gets historical data for either main or additional numbers."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def find_geometric_progressions(past_draws: List[List[int]]) -> Counter:
    """Finds geometric progressions in past draws."""
    progressions = Counter()
    for draw in past_draws:
        sorted_draw = sorted(set(draw))
        n = len(sorted_draw)
        if n < 3:
            continue

        for i in range(n - 2):
            a = sorted_draw[i]
            for j in range(i + 1, n - 1):
                b = sorted_draw[j]
                if a == 0:
                    continue
                ratio = b / a
                if ratio <= 0:
                    continue
                for k in range(j + 1, n):
                    c = sorted_draw[k]
                    if is_geometric_progression(a, b, c):
                        progression = (a, b, c)
                        progressions[progression] += 1
    return progressions


def is_geometric_progression(a: int, b: int, c: int) -> bool:
    """Checks if three numbers form a geometric progression."""
    return a != 0 and b != 0 and b ** 2 == a * c


def generate_numbers_from_progressions(
        progressions: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> Set[int]:
    """Generates numbers from geometric progressions."""
    predicted_numbers = set()
    top_progressions = progressions.most_common(3)

    for progression, _ in top_progressions:
        predicted_numbers.update(progression)
        extend_progression(predicted_numbers, progression, min_num, max_num)
        if len(predicted_numbers) >= required_numbers:
            break

    return predicted_numbers


def extend_progression(
        predicted_numbers: Set[int],
        progression: Tuple[int, int, int],
        min_num: int,
        max_num: int
) -> None:
    """Extends a geometric progression forward and backward."""
    a, b, c = progression
    ratio = b / a

    if ratio <= 0 or ratio == 1:
        return

    # Forward extension
    next_number = c * ratio
    while min_num <= next_number <= max_num and next_number.is_integer():
        predicted_numbers.add(int(next_number))
        next_number *= ratio

    # Backward extension
    prev_number = a / ratio
    while min_num <= prev_number <= max_num and prev_number.is_integer():
        predicted_numbers.add(int(prev_number))
        prev_number /= ratio