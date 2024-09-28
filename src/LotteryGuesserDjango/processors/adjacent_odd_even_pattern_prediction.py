from collections import Counter

from algorithms.models import lg_lottery_winner_number

from collections import Counter
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    adjacency_counter = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)  # Ensure the numbers are in order
        for i in range(len(sorted_draw) - 1):
            if sorted_draw[i] % 2 != sorted_draw[i + 1] % 2:  # Checking for odd-even adjacency
                adjacency_counter[sorted_draw[i]] += 1
                adjacency_counter[sorted_draw[i + 1]] += 1

    # Select more numbers than needed to ensure we have enough after filtering
    most_common_adjacent_numbers = [num for num, _ in
                                    adjacency_counter.most_common(lottery_type_instance.pieces_of_draw_numbers * 2)]

    # Filter numbers to ensure they're within the valid range
    valid_numbers = [num for num in most_common_adjacent_numbers
                     if lottery_type_instance.min_number <= num <= lottery_type_instance.max_number]

    # If we don't have enough numbers, add random numbers within the range
    while len(valid_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_num = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if new_num not in valid_numbers:
            valid_numbers.append(new_num)

    # Return only the required number of numbers
    return sorted(valid_numbers[:lottery_type_instance.pieces_of_draw_numbers])
