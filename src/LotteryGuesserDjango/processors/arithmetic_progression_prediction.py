import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    progression_differences = Counter()

    for draw in past_draws:
        for i in range(len(draw) - 1):
            difference = draw[i + 1] - draw[i]
            progression_differences[difference] += 1

    most_common_difference = progression_differences.most_common(1)[0][0]
    start_number = random.choice(range(lottery_type_instance.min_number, lottery_type_instance.max_number))
    predicted_numbers = [start_number + i * most_common_difference for i in range(lottery_type_instance.pieces_of_draw_numbers)]

    return predicted_numbers
