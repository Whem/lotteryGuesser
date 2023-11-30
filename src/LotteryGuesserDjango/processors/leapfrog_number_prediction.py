import random

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    last_draw = sorted(past_draws[-1]) if past_draws else []

    predicted_numbers = set()
    for number in last_draw:
        leapfrog_number = number + 2
        if leapfrog_number <= lottery_type_instance.max_number:
            predicted_numbers.add(leapfrog_number)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(predicted_numbers)