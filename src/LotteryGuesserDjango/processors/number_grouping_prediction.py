import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    number_counter = Counter()
    for draw in past_draws:
        number_counter.update(draw)

    grouped_numbers = {}
    for number, count in number_counter.items():
        grouped_numbers.setdefault(count, []).append(number)

    selected_numbers = set()
    while len(selected_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        group = random.choice(list(grouped_numbers.values()))
        selected_numbers.add(random.choice(group))

    return sorted(selected_numbers)