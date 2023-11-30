from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    pair_affinity = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            for j in range(i + 1, len(sorted_draw)):
                pair_affinity[(sorted_draw[i], sorted_draw[j])] += 1

    most_common_pairs = [pair for pair, _ in pair_affinity.most_common()]
    predicted_numbers = set()
    for pair in most_common_pairs:
        predicted_numbers.update(pair)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]