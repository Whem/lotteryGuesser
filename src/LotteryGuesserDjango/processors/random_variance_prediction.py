import random

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    all_numbers = [number for draw in past_draws for number in draw]
    mean_number = sum(all_numbers) / len(all_numbers)
    variance = max(all_numbers) - min(all_numbers)

    predicted_numbers = set()
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        random_number = int(mean_number + random.uniform(-variance, variance))
        if lottery_type_instance.min_number <= random_number <= lottery_type_instance.max_number:
            predicted_numbers.add(random_number)

    return sorted(predicted_numbers)