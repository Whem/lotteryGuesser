from collections import Counter
from itertools import combinations

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    pair_counter = Counter()

    for draw in past_draws:
        pairs = combinations(sorted(draw), 2)
        pair_counter.update(pairs)

    rarest_pairs = [pair for pair, count in pair_counter.items() if count == pair_counter.most_common()[-1][1]]
    selected_numbers = set()
    for pair in rarest_pairs:
        selected_numbers.update(pair)
        if len(selected_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    return sorted(selected_numbers)[:lottery_type_instance.pieces_of_draw_numbers]