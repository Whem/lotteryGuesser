# seasonal_trend_prediction.py

import random
import datetime
from collections import Counter
from typing import List, Tuple
import numpy as np
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Seasonal Trend Prediction.

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
    Generates a list of lottery numbers using Seasonal Trend Prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Get current week number
    today = datetime.date.today()
    current_week = today.isocalendar()[1]

    # Define seasons based on week numbers
    if 1 <= current_week <= 13:
        season_weeks = list(range(1, 14))  # Winter
    elif 14 <= current_week <= 26:
        season_weeks = list(range(14, 27))  # Spring
    elif 27 <= current_week <= 39:
        season_weeks = list(range(27, 40))  # Summer
    else:
        season_weeks = list(range(40, 53))  # Autumn

    # Retrieve past winning numbers during the same season
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance,
        lottery_type_number_week__in=season_weeks
    ).values_list(number_field, flat=True)

    past_draws = [
        [int(num) for num in draw] for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 20:
        # If not enough data, return the smallest 'total_numbers' numbers
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    # Count frequency of each number during the current season
    number_counter = Counter()
    for draw in past_draws:
        for number in draw:
            if min_num <= number <= max_num:
                number_counter[number] += 1

    # Select numbers based on frequency
    if number_counter:
        # Sort numbers by increasing frequency (least frequent first)
        all_numbers = list(range(min_num, max_num + 1))
        numbers_by_frequency = sorted(all_numbers, key=lambda x: number_counter.get(x, 0))
        selected_numbers = numbers_by_frequency[:total_numbers]
    else:
        # If no frequency data, select random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)

    # Ensure unique numbers and correct count
    selected_numbers = list(set(selected_numbers))
    if len(selected_numbers) < total_numbers:
        remaining_numbers = list(set(range(min_num, max_num + 1)) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])
    elif len(selected_numbers) > total_numbers:
        selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the numbers
    selected_numbers.sort()
    return selected_numbers
