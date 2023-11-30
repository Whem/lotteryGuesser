import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    number_counter = Counter()
    for draw in past_draws:
        number_counter.update(draw)

    half_draw_size = lottery_type_instance.pieces_of_draw_numbers // 2
    hot_numbers = [num for num, _ in number_counter.most_common(half_draw_size)]
    cold_numbers = [num for num, _ in number_counter.most_common()[:-half_draw_size-1:-1]]

    selected_numbers = set(hot_numbers + cold_numbers)
    while len(selected_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        selected_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(selected_numbers)