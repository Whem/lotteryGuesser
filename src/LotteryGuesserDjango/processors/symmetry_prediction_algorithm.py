from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    def is_symmetrical(numbers):
        return numbers == numbers[::-1]

    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    symmetry_counter = Counter()

    for draw in past_draws:
        if is_symmetrical(sorted(draw)):
            symmetry_counter.update(draw)

    most_common_symmetrical_numbers = [num for num, _ in symmetry_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(most_common_symmetrical_numbers)