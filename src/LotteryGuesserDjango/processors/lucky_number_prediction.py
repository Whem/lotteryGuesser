from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    luckiness_counter = Counter()

    for draw in past_draws:
        luckiness_counter.update(draw)

    lucky_numbers = [num for num, _ in luckiness_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(lucky_numbers)