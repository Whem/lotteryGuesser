from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    gap_counter = Counter()
    last_seen = {}

    for draw in past_draws:
        for number in draw:
            if number in last_seen:
                gap = len(past_draws) - last_seen[number]
                gap_counter[number] += gap
            last_seen[number] = len(past_draws)

    most_common_gaps = [num for num, _ in gap_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(most_common_gaps)