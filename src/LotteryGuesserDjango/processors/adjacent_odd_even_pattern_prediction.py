#adjacent_odd_even_pattern_prediction.py
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate main and additional numbers based on adjacent odd-even pattern prediction.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    adjacency_counter = Counter()

    # Calculate adjacency frequency based on odd-even pairs in past draws
    for draw in past_draws:
        sorted_draw = sorted(draw) if isinstance(draw, list) else []
        for i in range(len(sorted_draw) - 1):
            if sorted_draw[i] % 2 != sorted_draw[i + 1] % 2:
                adjacency_counter[sorted_draw[i]] += 1
                adjacency_counter[sorted_draw[i + 1]] += 1

    # Most common numbers for main set
    most_common_adjacent_numbers = [num for num, _ in
                                    adjacency_counter.most_common(lottery_type_instance.pieces_of_draw_numbers * 2)]
    valid_main_numbers = [num for num in most_common_adjacent_numbers if
                          lottery_type_instance.min_number <= num <= lottery_type_instance.max_number]

    # Fill main numbers, adding random numbers if needed
    main_numbers = set(valid_main_numbers[:lottery_type_instance.pieces_of_draw_numbers])
    while len(main_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_num = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        main_numbers.add(new_num)

    # Additional numbers if required by the lottery type
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_adjacent_numbers = [num for num, _ in adjacency_counter.most_common(
            lottery_type_instance.additional_numbers_count * 2)]
        valid_additional_numbers = [num for num in additional_adjacent_numbers if
                                    lottery_type_instance.additional_min_number <= num <= lottery_type_instance.additional_max_number]

        additional_numbers = set(valid_additional_numbers[:lottery_type_instance.additional_numbers_count])
        while len(additional_numbers) < lottery_type_instance.additional_numbers_count:
            new_num = random.randint(lottery_type_instance.additional_min_number,
                                     lottery_type_instance.additional_max_number)
            additional_numbers.add(new_num)

    return sorted(main_numbers), sorted(additional_numbers)
