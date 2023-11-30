import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    distance_counter = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)
        distances = [sorted_draw[i + 1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
        distance_counter.update(distances)

    average_distance = sum(distance_counter.elements()) / len(distance_counter)
    start_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
    predicted_numbers = set([start_number])

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        start_number += int(average_distance)
        if start_number <= lottery_type_instance.max_number:
            predicted_numbers.add(start_number)

    return sorted(predicted_numbers)