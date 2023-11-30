import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    min_number = lottery_type_instance.min_number
    max_number = lottery_type_instance.max_number
    num_draws = lottery_type_instance.pieces_of_draw_numbers

    odd_counter = Counter()
    even_counter = Counter()

    for draw in past_draws:
        for number in draw:
            if number % 2 == 0:
                even_counter[number] += 1
            else:
                odd_counter[number] += 1

    odd_common = [num for num, _ in odd_counter.most_common(num_draws // 2)]
    even_common = [num for num, _ in even_counter.most_common(num_draws // 2 + num_draws % 2)]

    selected_numbers = set(odd_common + even_common)
    while len(selected_numbers) < num_draws:
        selected_numbers.add(random.randint(min_number, max_number))

    return sorted(selected_numbers)