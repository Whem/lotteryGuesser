from collections import Counter
from itertools import combinations

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    pair_gap_counter = Counter()

    for draw in past_draws:
        for pair in combinations(sorted(draw), 2):
            gap = pair[1] - pair[0]
            pair_gap_counter[gap] += 1

    most_common_gaps = [gap for gap, _ in pair_gap_counter.most_common()]
    predicted_numbers = set()
    for gap in most_common_gaps:
        for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number - gap):
            predicted_numbers.add(number)
            predicted_numbers.add(number + gap)
            if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                break

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]
