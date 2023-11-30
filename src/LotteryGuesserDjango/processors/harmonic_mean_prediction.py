import random

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    all_numbers = [number for draw in past_draws for number in draw]

    harmonic_mean = len(all_numbers) / sum(1 / number for number in all_numbers if number != 0)
    predicted_numbers = set()

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        number = int(harmonic_mean + random.uniform(-harmonic_mean, harmonic_mean))
        if lottery_type_instance.min_number <= number <= lottery_type_instance.max_number:
            predicted_numbers.add(number)

    return sorted(predicted_numbers)
