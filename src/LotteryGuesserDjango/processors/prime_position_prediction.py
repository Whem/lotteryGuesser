from collections import Counter

import sympy

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    prime_position_counter = Counter()

    for draw in past_draws:
        for position, number in enumerate(draw, start=1):
            if sympy.isprime(position):
                prime_position_counter[number] += 1

    prime_position_common = [num for num, _ in prime_position_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(prime_position_common)
