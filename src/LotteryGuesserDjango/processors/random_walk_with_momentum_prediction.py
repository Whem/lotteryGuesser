import random

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    if not past_draws:
        return [random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number) for _ in range(lottery_type_instance.pieces_of_draw_numbers)]

    last_draw = sorted(past_draws[-1])
    momentum = [last_draw[i + 1] - last_draw[i] for i in range(len(last_draw) - 1)]
    predicted_numbers = set(last_draw)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        next_number = last_draw[-1] + random.choice(momentum)
        if lottery_type_instance.min_number <= next_number <= lottery_type_instance.max_number:
            predicted_numbers.add(next_number)
            last_draw.append(next_number)
            last_draw = sorted(last_draw)
            momentum.append(last_draw[-1] - last_draw[-2])

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]