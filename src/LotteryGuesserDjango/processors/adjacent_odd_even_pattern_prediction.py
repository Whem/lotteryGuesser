from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    adjacency_counter = Counter()

    for draw in past_draws:
        for i in range(len(draw) - 1):
            if draw[i] % 2 != draw[i + 1] % 2:  # Checking for odd-even adjacency
                adjacency_counter[draw[i]] += 1
                adjacency_counter[draw[i + 1]] += 1

    most_common_adjacent_numbers = [num for num, _ in adjacency_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(most_common_adjacent_numbers)
