import itertools
import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    permutation_cycles = Counter()

    for draw in past_draws:
        for cycle_length in range(2, lottery_type_instance.pieces_of_draw_numbers + 1):
            for cycle in itertools.permutations(draw, cycle_length):
                permutation_cycles[cycle] += 1

    most_common_cycle = permutation_cycles.most_common(1)[0][0]
    predicted_numbers = set(most_common_cycle)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(predicted_numbers)
