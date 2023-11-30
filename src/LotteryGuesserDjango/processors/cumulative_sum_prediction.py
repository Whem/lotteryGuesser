import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    cumulative_sums = [sum(draw) for draw in past_draws]
    common_cumulative_sum = Counter(cumulative_sums).most_common(1)[0][0]

    predicted_numbers = set()
    while sum(predicted_numbers) != common_cumulative_sum and len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))
        if sum(predicted_numbers) > common_cumulative_sum:
            predicted_numbers.clear()

    return sorted(predicted_numbers)