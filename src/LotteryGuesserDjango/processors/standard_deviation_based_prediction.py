import random

import numpy as np

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    all_numbers = [number for draw in past_draws for number in draw]

    mean = np.mean(all_numbers)
    std_dev = np.std(all_numbers)
    lower_bound = mean - std_dev
    upper_bound = mean + std_dev

    predicted_numbers = set()
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if lower_bound <= number <= upper_bound:
            predicted_numbers.add(number)

    return sorted(predicted_numbers)