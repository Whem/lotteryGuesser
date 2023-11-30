from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    def reduce_to_single_digit(num):
        return sum(int(digit) for digit in str(num)) % 9 or 9

    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    numerology_counter = Counter()

    for draw in past_draws:
        reduced_numbers = [reduce_to_single_digit(number) for number in draw]
        numerology_counter.update(reduced_numbers)

    most_common_numerology = [num for num, _ in numerology_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(most_common_numerology)