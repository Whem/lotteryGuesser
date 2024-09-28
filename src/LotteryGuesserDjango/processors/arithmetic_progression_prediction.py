import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number

import random
from collections import Counter
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    progression_differences = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)  # Ensure the numbers are in order
        for i in range(len(sorted_draw) - 1):
            difference = sorted_draw[i + 1] - sorted_draw[i]
            progression_differences[difference] += 1

    if not progression_differences:
        # If no differences found, use a random difference
        most_common_difference = random.randint(1, (
                    lottery_type_instance.max_number - lottery_type_instance.min_number) // lottery_type_instance.pieces_of_draw_numbers)
    else:
        most_common_difference = progression_differences.most_common(1)[0][0]

    valid_numbers = []
    attempts = 0
    max_attempts = 100  # Prevent infinite loop

    while len(valid_numbers) < lottery_type_instance.pieces_of_draw_numbers and attempts < max_attempts:
        start_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        predicted_numbers = [start_number + i * most_common_difference for i in
                             range(lottery_type_instance.pieces_of_draw_numbers)]

        # Filter valid numbers
        valid_numbers = [num for num in predicted_numbers
                         if lottery_type_instance.min_number <= num <= lottery_type_instance.max_number]
        attempts += 1

    # If we couldn't generate a valid sequence, fall back to random selection
    if len(valid_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        valid_numbers = random.sample(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
                                      lottery_type_instance.pieces_of_draw_numbers)

    return sorted(valid_numbers)
