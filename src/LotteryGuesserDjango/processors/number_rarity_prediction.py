from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    frequency_counter = Counter()

    for draw in past_draws:
        frequency_counter.update(draw)

    rare_numbers = [num for num, count in frequency_counter.items() if count == min(frequency_counter.values())]
    return sorted(rare_numbers)[:lottery_type_instance.pieces_of_draw_numbers]
