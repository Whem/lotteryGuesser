# interval_prediction.py
from typing import List, Dict, Tuple
from collections import defaultdict
from statistics import mean
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random

def get_numbers(lottery_type: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers based on interval analysis for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type,
        lottery_type.min_number,
        lottery_type.max_number,
        lottery_type.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type,
            lottery_type.additional_min_number,
            lottery_type.additional_max_number,
            lottery_type.additional_numbers_count,
            is_main=False
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    lottery_type: lg_lottery_type,
    min_num: int,
    max_num: int,
    required_numbers: int,
    is_main: bool
) -> List[int]:
    """
    Generate a set of numbers using interval analysis.
    """
    # Fetch past draws
    queryset = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).values_list(
        'lottery_type_number', flat=True
    )

    intervals = defaultdict(list)
    last_occurrence = {}
    current_draw = 0

    for numbers in queryset:
        if not isinstance(numbers, list):
            continue

        if is_main:
            # Extract main numbers within the specified range
            numbers = [num for num in numbers if min_num <= num <= max_num]
        else:
            # Assuming additional numbers are stored separately
            numbers = [num for num in numbers if min_num <= num <= max_num]

        for number in numbers:
            if number in last_occurrence:
                intervals[number].append(current_draw - last_occurrence[number])
            last_occurrence[number] = current_draw
        current_draw += 1

    # Calculate average intervals
    if intervals:
        average_intervals = {number: mean(interval_list) for number, interval_list in intervals.items()}
        # Sort numbers by shortest average interval (numbers that are "due" to appear)
        sorted_numbers = sorted(average_intervals, key=lambda x: (average_intervals[x], -x))
    else:
        # If no intervals data is available, use all possible numbers
        sorted_numbers = list(range(min_num, max_num + 1))

    predicted_numbers = sorted_numbers[:required_numbers]

    # Ensure we have enough numbers
    if len(predicted_numbers) < required_numbers:
        remaining_numbers = set(range(min_num, max_num + 1)) - set(predicted_numbers)
        predicted_numbers.extend(random.sample(remaining_numbers, required_numbers - len(predicted_numbers)))

    # Convert to standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    return predicted_numbers[:required_numbers]
