#geometric_pattern_prediction.py
import random
from typing import List, Tuple, Set
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on geometric progression patterns found in past draws.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two sorted lists:
      1. `main_numbers`: Predicted main lottery numbers.
      2. `additional_numbers`: Predicted additional lottery numbers (if applicable).
    """
    # Get past draws for main numbers
    main_numbers = get_numbers_for_type(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Get additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = get_numbers_for_type(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    # Return both lists as a tuple
    return main_numbers, additional_numbers



def get_numbers_for_type(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    required_numbers: int,
    is_main: bool
) -> List[int]:
    """Generate numbers for either main or additional numbers."""
    try:
        # Get past draws from JSONField
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        )

        # Convert QuerySet to list of numbers based on type
        past_draws = []
        for draw in past_draws_queryset:
            try:
                numbers = draw.lottery_type_number if is_main else draw.additional_numbers
                if isinstance(numbers, list) and numbers:  # Check if it's a non-empty list
                    past_draws.append(numbers)
            except (ValueError, TypeError, AttributeError):
                continue

        if not past_draws:
            return generate_random_numbers(min_num, max_num, required_numbers)

    except Exception as e:
        print(f"Error processing past draws: {e}")
        return generate_random_numbers(min_num, max_num, required_numbers)

    # Find geometric progressions in past draws
    progressions = find_geometric_progressions(past_draws)

    # Generate numbers from the most common progressions
    predicted_numbers = generate_numbers_from_progressions(
        progressions,
        min_num,
        max_num,
        required_numbers
    )

    # If we don't have enough numbers, fill with random ones
    while len(predicted_numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        predicted_numbers.add(new_number)

    # Return the required number of sorted numbers
    return sorted(list(predicted_numbers))[:required_numbers]


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers within given constraints."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))


def find_geometric_progressions(past_draws: List[List[int]]) -> Counter:
    """Find geometric progressions in past lottery draws."""
    progressions = Counter()
    for draw in past_draws:
        try:
            sorted_draw = sorted(set(draw))  # Remove duplicates
            n = len(sorted_draw)
            if n < 3:
                continue
            for i in range(n - 2):
                a = sorted_draw[i]
                for j in range(i + 1, n - 1):
                    b = sorted_draw[j]
                    if a == 0:
                        continue
                    try:
                        ratio = b / a
                        if ratio <= 0:
                            continue
                        for k in range(j + 1, n):
                            c = sorted_draw[k]
                            if is_geometric_progression(a, b, c):
                                progression = (a, b, c)
                                progressions[progression] += 1
                    except (ZeroDivisionError, TypeError):
                        continue
        except Exception:
            continue
    return progressions


def is_geometric_progression(a: int, b: int, c: int) -> bool:
    """Check if three numbers form a geometric progression."""
    try:
        if a == 0 or b == 0:
            return False
        return abs((b * b) - (a * c)) < 0.000001  # Using small epsilon for float comparison
    except (TypeError, ValueError):
        return False


def generate_numbers_from_progressions(
    progressions: Counter,
    min_num: int,
    max_num: int,
    required_numbers: int
) -> Set[int]:
    """Generate predicted numbers from geometric progressions."""
    predicted_numbers = set()
    top_progressions = progressions.most_common(3)

    for progression, _ in top_progressions:
        predicted_numbers.update(progression)

        # Extend the progression
        extend_progression(predicted_numbers, progression, min_num, max_num)

        # Stop if we have enough numbers
        if len(predicted_numbers) >= required_numbers:
            break

    return predicted_numbers


def extend_progression(
    predicted_numbers: Set[int],
    progression: Tuple[int, int, int],
    min_num: int,
    max_num: int
) -> None:
    """Extend a geometric progression forward and backward."""
    try:
        a, b, c = progression
        if a == 0:
            return

        ratio = b / a

        if ratio <= 0 or ratio == 1:
            return

        # Extend forward
        next_number = c * ratio
        while min_num <= next_number <= max_num and abs(next_number - round(next_number)) < 0.000001:
            predicted_numbers.add(round(next_number))
            next_number *= ratio

        # Extend backward
        prev_number = a / ratio
        while min_num <= prev_number <= max_num and abs(prev_number - round(prev_number)) < 0.000001:
            predicted_numbers.add(round(prev_number))
            prev_number /= ratio
    except (TypeError, ValueError, ZeroDivisionError):
        return