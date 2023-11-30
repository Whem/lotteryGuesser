import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    range_counter = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)
        range_min, range_max = sorted_draw[0], sorted_draw[-1]
        range_counter[(range_min, range_max)] += 1

    most_common_range = max(range_counter, key=range_counter.get)
    predicted_numbers = set(random.randint(most_common_range[0], most_common_range[1]) for _ in range(lottery_type_instance.pieces_of_draw_numbers))

    return sorted(predicted_numbers)