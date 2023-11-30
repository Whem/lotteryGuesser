from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    reverse_order_counter = Counter()

    for draw in past_draws:
        reverse_draw = tuple(sorted(draw, reverse=True))
        reverse_order_counter[reverse_draw] += 1

    most_common_reverse_order = reverse_order_counter.most_common(1)[0][0]
    return sorted(most_common_reverse_order[:lottery_type_instance.pieces_of_draw_numbers])
