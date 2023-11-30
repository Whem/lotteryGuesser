from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    position_counters = [Counter() for _ in range(lottery_type_instance.pieces_of_draw_numbers)]

    for draw in past_draws:
        for position, number in enumerate(sorted(draw)):
            position_counters[position][number] += 1

    predicted_numbers = [counter.most_common(1)[0][0] for counter in position_counters]
    return sorted(predicted_numbers)