import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    range_counter = Counter()
    for draw in past_draws:
        for i in range(len(draw)):
            for j in range(i+1, len(draw)):
                range_counter[(draw[i], draw[j])] += 1

    most_common_ranges = [rng for rng, _ in range_counter.most_common()]
    selected_numbers = set()
    while len(selected_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        selected_range = random.choice(most_common_ranges)
        for num in range(selected_range[0], selected_range[1] + 1):
            selected_numbers.add(num)
            if len(selected_numbers) == lottery_type_instance.pieces_of_draw_numbers:
                break

    return sorted(selected_numbers)