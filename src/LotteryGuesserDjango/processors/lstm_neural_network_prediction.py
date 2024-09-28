import numpy as np
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:50].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 10:
        return sorted(np.random.choice(range(min_num, max_num + 1), total_numbers, replace=False))

    # Calculate moving average and standard deviation
    flat_past_draws = [num for draw in past_draws for num in draw]
    moving_avg = np.mean(flat_past_draws)
    moving_std = np.std(flat_past_draws)

    # Generate numbers based on the moving average and standard deviation
    predicted_numbers = set()
    while len(predicted_numbers) < total_numbers:
        number = int(np.random.normal(moving_avg, moving_std))
        if min_num <= number <= max_num and number not in predicted_numbers:
            predicted_numbers.add(number)

    return sorted(list(predicted_numbers))