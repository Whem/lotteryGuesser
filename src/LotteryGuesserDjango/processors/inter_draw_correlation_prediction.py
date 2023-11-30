import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    number_correlations = Counter()

    for i in range(len(past_draws) - 1):
        for number in past_draws[i]:
            for next_number in past_draws[i + 1]:
                number_correlations[(number, next_number)] += 1

    most_common_correlation = number_correlations.most_common(1)[0][0]
    predicted_numbers = set(most_common_correlation)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(predicted_numbers)