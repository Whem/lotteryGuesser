import random

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    def is_factor_or_multiple(x, y):
        return x % y == 0 or y % x == 0

    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    last_draw = sorted(past_draws[-1]) if past_draws else []
    predicted_numbers = set(last_draw)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if any(is_factor_or_multiple(number, prev_number) for prev_number in predicted_numbers):
            predicted_numbers.add(number)

    return sorted(predicted_numbers)
