# symmetry_prediction_algorithm.py

import random
from typing import List, Tuple, Set, Dict
from collections import defaultdict, Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def is_symmetric(number: int) -> bool:
    """
    Determines if a given number is symmetric (palindromic).

    Parameters:
    - number: The number to check for symmetry.

    Returns:
    - True if the number is symmetric, False otherwise.
    """
    number_str = str(number)
    return number_str == number_str[::-1]


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on the symmetry prediction algorithm.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    symmetric numbers in past lottery draws. It prioritizes selecting the most frequently occurring
    symmetric numbers and fills any remaining slots with other symmetric or random numbers as needed.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance=lottery_type_instance,
        number_field='lottery_type_number',
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count)
        )

    return main_numbers, additional_numbers


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers based on the symmetry prediction algorithm.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to identify symmetric numbers, prioritizes
    them based on frequency, and fills any remaining slots with other symmetric or random numbers.

    Parameters:
    - lottery_type_instance: The lottery type instance.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers
                    ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Number of numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    try:
        # Retrieve past winning numbers
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list(number_field, flat=True)

        past_draws = [
            draw for draw in past_draws_queryset
            if isinstance(draw, list) and len(draw) == total_numbers
        ]

        if len(past_draws) < 10:
            # If not enough past draws, generate random numbers
            selected_numbers = generate_random_numbers(min_num, max_num, total_numbers)
            return selected_numbers

        # Identify all symmetric numbers in the range
        symmetrical_numbers = [num for num in range(min_num, max_num + 1) if is_symmetric(num)]

        # Count frequency of symmetric numbers in past draws
        symmetry_counter = Counter()
        for draw in past_draws:
            for number in draw:
                if is_symmetric(number):
                    symmetry_counter[number] += 1

        # Sort symmetrical numbers by frequency in descending order
        sorted_symmetrical_numbers = [num for num, _ in symmetry_counter.most_common()]

        selected_numbers = sorted_symmetrical_numbers.copy()

        # Add remaining symmetrical numbers if needed
        if len(selected_numbers) < total_numbers:
            remaining_symmetrical_numbers = list(set(symmetrical_numbers) - set(selected_numbers))
            random.shuffle(remaining_symmetrical_numbers)
            selected_numbers.extend(remaining_symmetrical_numbers[:total_numbers - len(selected_numbers)])

        # If still not enough, fill with random symmetric numbers
        if len(selected_numbers) < total_numbers:
            all_symmetrical = set(symmetrical_numbers)
            remaining_numbers = list(all_symmetrical - set(selected_numbers))
            random.shuffle(remaining_numbers)
            selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

        # If still not enough, fill with any random numbers within the range
        if len(selected_numbers) < total_numbers:
            all_numbers = set(range(min_num, max_num + 1))
            remaining_numbers = list(all_numbers - set(selected_numbers))
            random.shuffle(remaining_numbers)
            selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

        # Ensure the correct number of numbers and sort them
        selected_numbers = selected_numbers[:total_numbers]
        selected_numbers.sort()
        return selected_numbers

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in generate_numbers: {str(e)}")
        # Fallback to random number generation
        return generate_random_numbers(min_num, max_num, total_numbers)


def generate_random_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generates a sorted list of unique random numbers within the specified range.

    Parameters:
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Number of numbers to generate.

    Returns:
    - A sorted list of randomly generated lottery numbers.
    """
    try:
        numbers = set()
        while len(numbers) < total_numbers:
            num = random.randint(min_num, max_num)
            numbers.add(num)
        return sorted(list(numbers))
    except Exception as e:
        print(f"Error in generate_random_numbers: {str(e)}")
        # As a last resort, return a sequential list
        return list(range(min_num, min_num + total_numbers))


# Optional: Additional helper functions can be added here if needed.
