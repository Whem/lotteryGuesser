from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    adjacency_counter = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            adjacency_counter[(sorted_draw[i], sorted_draw[i + 1])] += 1

    most_common_adjacencies = [pair for pair, _ in adjacency_counter.most_common()]
    selected_numbers = set()
    for pair in most_common_adjacencies:
        selected_numbers.update(pair)
        if len(selected_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    return sorted(selected_numbers)[:lottery_type_instance.pieces_of_draw_numbers]