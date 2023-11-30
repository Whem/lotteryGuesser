import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    progressions = Counter()

    for draw in past_draws:
        for i in range(len(draw) - 2):
            if draw[i] * draw[i + 2] == draw[i + 1] ** 2:
                progression = (draw[i], draw[i + 1], draw[i + 2])
                progressions[progression] += 1

    most_common_progression = progressions.most_common(1)[0][0]
    predicted_numbers = set(most_common_progression)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(predicted_numbers)
