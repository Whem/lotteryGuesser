# random_walk_prediction.py
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on random walk prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance=lottery_type_instance,
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers,
        number_field='lottery_type_number'
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance=lottery_type_instance,
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count,
            number_field='additional_numbers'
        )

    return main_numbers, additional_numbers

def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_numbers: int,
    number_field: str
) -> List[int]:
    """
    Generates a set of lottery numbers based on random walk prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers ('lottery_type_number' or 'additional_numbers').

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve the most recent winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id').values_list(number_field, flat=True)

    if past_draws.exists():
        # Start from the most recent draw if usable
        last_draw = past_draws.first()
        if isinstance(last_draw, list) and len(last_draw) == total_numbers:
            current_numbers = sorted(last_draw)
        else:
            # Deterministic fallback when last draw unusable
            current_numbers = deterministic_fallback(min_num, max_num, total_numbers)
    else:
        # Deterministic fallback when no past draws
        current_numbers = deterministic_fallback(min_num, max_num, total_numbers)

    # Perform a deterministic walk using a fixed move pattern [0, +1, -1, 0, +1, -1, ...]
    predicted_numbers = []
    move_pattern = [0, 1, -1]
    for idx, num in enumerate(current_numbers):
        move = move_pattern[idx % len(move_pattern)]
        new_num = num + move

        # Ensure the new number is within bounds
        if new_num < min_num:
            new_num = min_num
        elif new_num > max_num:
            new_num = max_num

        predicted_numbers.append(new_num)

    # Ensure unique numbers
    predicted_unique = []
    seen = set()
    for n in predicted_numbers:
        if n not in seen:
            predicted_unique.append(n)
            seen.add(n)

    # If not enough numbers, fill deterministically with smallest remaining numbers
    if len(predicted_unique) < total_numbers:
        needed = total_numbers - len(predicted_unique)
        for candidate in range(min_num, max_num + 1):
            if candidate not in seen:
                predicted_unique.append(candidate)
                seen.add(candidate)
                if len(predicted_unique) >= total_numbers:
                    break
    elif len(predicted_unique) > total_numbers:
        predicted_unique = predicted_unique[:total_numbers]

    # Sort and return the selected numbers
    predicted_unique.sort()
    return predicted_unique


def deterministic_fallback(min_num: int, max_num: int, count: int) -> List[int]:
    """Return the first 'count' numbers in range [min_num, max_num] deterministically."""
    span = max_num - min_num + 1
    if count <= 0 or span <= 0:
        return []
    if count >= span:
        return list(range(min_num, max_num + 1))[:count]
    return list(range(min_num, min_num + count))
