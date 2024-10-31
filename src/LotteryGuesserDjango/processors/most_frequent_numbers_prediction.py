# most_frequent_numbers_prediction.py

from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers based on the most frequently occurring numbers from past draws.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_most_frequent_numbers(
        lottery_type_instance,
        min_number=int(lottery_type_instance.min_number),
        max_number=int(lottery_type_instance.max_number),
        pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_most_frequent_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.additional_min_number),
            max_number=int(lottery_type_instance.additional_max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_most_frequent_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    numbers_field: str
) -> List[int]:
    """
    Generate the most frequently occurring numbers from past draws.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(numbers_field, flat=True)

    past_draws = [
        draw for draw in past_draws_queryset
        if isinstance(draw, list)
    ]

    if not past_draws:
        # If insufficient data, return the first 'pieces_of_draw_numbers' numbers
        selected_numbers = list(range(min_number, min_number + pieces_of_draw_numbers))
        return selected_numbers

    # Flatten the list to get all numbers
    all_numbers = [number for draw in past_draws for number in draw]

    # Count the frequency of each number
    number_counts = Counter(all_numbers)

    # Get the numbers sorted by frequency (most common first)
    most_common_numbers = [num for num, count in number_counts.most_common()]

    # Select the first 'pieces_of_draw_numbers' numbers
    selected_numbers = most_common_numbers[:pieces_of_draw_numbers]

    # If not enough numbers, fill in the missing numbers
    if len(selected_numbers) < pieces_of_draw_numbers:
        remaining_numbers = [
            num for num in range(min_number, max_number + 1) if num not in selected_numbers
        ]
        selected_numbers.extend(remaining_numbers[:pieces_of_draw_numbers - len(selected_numbers)])

    # Convert numbers to standard Python int
    selected_numbers = [int(num) for num in selected_numbers]

    return selected_numbers
