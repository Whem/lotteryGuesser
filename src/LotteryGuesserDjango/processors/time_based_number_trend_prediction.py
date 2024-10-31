# time_based_number_trend_prediction.py

import random
import math
import datetime
from collections import Counter
from typing import List, Tuple, Set, Dict
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
    Generates lottery numbers based on time-based number trend prediction.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    the frequency trends of symmetric numbers over recent and previous weeks. It prioritizes selecting
    numbers that have increased in frequency compared to earlier periods and fills any remaining slots
    with random numbers within the symmetric range.

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
    Generates a list of lottery numbers based on time-based number trend prediction.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It analyzes past draws to identify symmetric numbers, calculates
    their frequency trends over recent and previous weeks, prioritizes those with increasing trends,
    and fills any remaining slots with random numbers within the symmetric range.

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
        # Define the number of recent weeks to analyze
        recent_weeks = 10  # Adjust as needed

        # Get current year and week
        current_date = datetime.date.today()
        current_year, current_week, _ = current_date.isocalendar()

        # Initialize counters for recent and previous periods
        recent_numbers = Counter()
        previous_numbers = Counter()

        # Collect numbers from recent weeks
        for week_offset in range(recent_weeks):
            # Calculate target week and year
            target_week = current_week - week_offset
            target_year = current_year

            # Adjust for year change if necessary
            while target_week < 1:
                target_year -= 1
                last_week_of_year = datetime.date(target_year, 12, 28).isocalendar()[1]
                target_week += last_week_of_year

            # Retrieve draws for the target week and year
            draws = lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance,
                lottery_type_number_year=target_year,
                lottery_type_number_week=target_week
            ).values_list('lottery_type_number', flat=True)

            for draw in draws:
                if isinstance(draw, list):
                    for number in draw:
                        if isinstance(number, int) and is_symmetric(number):
                            recent_numbers[number] += 1

        # Collect numbers from previous weeks (same number of weeks)
        for week_offset in range(recent_weeks, recent_weeks * 2):
            # Calculate target week and year
            target_week = current_week - week_offset
            target_year = current_year

            # Adjust for year change if necessary
            while target_week < 1:
                target_year -= 1
                last_week_of_year = datetime.date(target_year, 12, 28).isocalendar()[1]
                target_week += last_week_of_year

            # Retrieve draws for the target week and year
            draws = lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance,
                lottery_type_number_year=target_year,
                lottery_type_number_week=target_week
            ).values_list('lottery_type_number', flat=True)

            for draw in draws:
                if isinstance(draw, list):
                    for number in draw:
                        if isinstance(number, int) and is_symmetric(number):
                            previous_numbers[number] += 1

        # Calculate trend scores for each number
        trend_scores = {}
        all_numbers = set(recent_numbers.keys()).union(previous_numbers.keys())
        for num in all_numbers:
            recent_freq = recent_numbers.get(num, 0)
            previous_freq = previous_numbers.get(num, 0)
            if previous_freq == 0:
                trend = recent_freq
            else:
                trend = recent_freq / previous_freq
            trend_scores[num] = trend

        # Sort numbers based on trend scores in descending order
        sorted_numbers = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)

        # Select the top numbers based on the required pieces_of_draw_numbers
        selected_numbers = [num for num, trend in sorted_numbers if trend > 1]

        # Add remaining symmetrical numbers if needed
        if len(selected_numbers) < total_numbers:
            symmetrical_numbers = [num for num in range(min_num, max_num + 1) if is_symmetric(num)]
            remaining_symmetrical_numbers = list(set(symmetrical_numbers) - set(selected_numbers))
            random.shuffle(remaining_symmetrical_numbers)
            selected_numbers.extend(remaining_symmetrical_numbers[:total_numbers - len(selected_numbers)])

        # If still not enough, fill with random symmetric numbers
        if len(selected_numbers) < total_numbers:
            all_symmetrical = set(num for num in range(min_num, max_num + 1) if is_symmetric(num))
            remaining_numbers = list(all_symmetrical - set(selected_numbers))
            random.shuffle(remaining_numbers)
            selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

        # If still not enough, fill with any random numbers within the range
        if len(selected_numbers) < total_numbers:
            all_numbers_set = set(range(min_num, max_num + 1))
            remaining_numbers = list(all_numbers_set - set(selected_numbers))
            random.shuffle(remaining_numbers)
            selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

        # Ensure we have the correct number of numbers and sort them
        selected_numbers = selected_numbers[:total_numbers]
        selected_numbers.sort()
        return selected_numbers

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in generate_numbers: {str(e)}")
        # Fall back to random number generation
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
