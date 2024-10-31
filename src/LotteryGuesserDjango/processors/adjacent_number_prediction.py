#adjacent_number_prediction.py
from collections import Counter
from algorithms.models import lg_lottery_winner_number
import random
from typing import List, Tuple


def get_numbers(lottery_type_instance) -> Tuple[List[int], List[int]]:
    """
    Generate main and additional numbers based on adjacent number prediction.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Calculate frequency of each number in past draws
    number_frequency = Counter()
    for draw in past_draws:
        if isinstance(draw, list):
            number_frequency.update(draw)

    # Select most common numbers for the main numbers
    most_common_numbers = [num for num, _ in
                           number_frequency.most_common(lottery_type_instance.pieces_of_draw_numbers * 2)]

    # Randomly select main numbers from the most common numbers
    main_numbers = set()
    while len(main_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        if most_common_numbers:
            main_numbers.add(random.choice(most_common_numbers))
        else:
            # If there are no more common numbers, select from the full range
            main_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    # Generate additional numbers if the lottery type requires it
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_common_numbers = [num for num, _ in
                                     number_frequency.most_common(lottery_type_instance.additional_numbers_count * 2)]
        additional_numbers = set()

        while len(additional_numbers) < lottery_type_instance.additional_numbers_count:
            if additional_common_numbers:
                additional_numbers.add(random.choice(additional_common_numbers))
            else:
                additional_numbers.add(random.randint(lottery_type_instance.additional_min_number,
                                                      lottery_type_instance.additional_max_number))

    return sorted(main_numbers), sorted(additional_numbers)
