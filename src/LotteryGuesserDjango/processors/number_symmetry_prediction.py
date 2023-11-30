from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    symmetry_counter = Counter()

    for draw in past_draws:
        symmetrical_numbers = [number for number in draw if str(number) == str(number)[::-1]]
        symmetry_counter.update(symmetrical_numbers)

    symmetrical_trends = [num for num, _ in symmetry_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(symmetrical_trends)
