import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    wave_counter = Counter()

    for index, draw in enumerate(past_draws):
        for number in draw:
            wave_pattern = (number, index % 2)
            wave_counter[wave_pattern] += 1

    most_common_wave = wave_counter.most_common(1)[0][0][0]
    predicted_numbers = set([most_common_wave])

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(predicted_numbers)

