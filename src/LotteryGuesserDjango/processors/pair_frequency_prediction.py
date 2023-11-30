from itertools import combinations
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    min_number = lottery_type_instance.min_number
    max_number = lottery_type_instance.max_number
    num_draws = lottery_type_instance.pieces_of_draw_numbers

    pair_counter = Counter()
    for draw in past_draws:
        pairs = combinations(sorted(draw), 2)
        pair_counter.update(pairs)

    most_common_pairs = [pair for pair, _ in pair_counter.most_common(num_draws)]
    selected_numbers = set()
    for pair in most_common_pairs:
        selected_numbers.update(pair)

    # Ensure we only return the required number of draws
    return sorted(selected_numbers)[:num_draws]