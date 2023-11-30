import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    delta_counter = Counter()
    for draw in past_draws:
        sorted_draw = sorted(draw)
        deltas = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
        delta_counter.update(deltas)

    most_common_deltas = [delta for delta, _ in delta_counter.most_common()]
    selected_numbers = set()
    while len(selected_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        start_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        for delta in most_common_deltas:
            next_number = start_number + delta
            if lottery_type_instance.min_number <= next_number <= lottery_type_instance.max_number:
                selected_numbers.add(next_number)
                start_number = next_number
                if len(selected_numbers) == lottery_type_instance.pieces_of_draw_numbers:
                    break

    return sorted(selected_numbers)