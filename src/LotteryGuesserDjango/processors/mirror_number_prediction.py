from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    mirror_counter = Counter()

    for draw in past_draws:
        for number in draw:
            mirror_number = int(str(number)[::-1])
            mirror_counter[mirror_number] += 1

    most_common_mirrors = [num for num, _ in mirror_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(most_common_mirrors)