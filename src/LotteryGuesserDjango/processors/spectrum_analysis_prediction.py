import numpy as np
from scipy import fft
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 20:
        return sorted(np.random.choice(range(lottery_type_instance.min_number,
                                             lottery_type_instance.max_number + 1),
                                       lottery_type_instance.pieces_of_draw_numbers,
                                       replace=False))

    flat_past_draws = [num for draw in past_draws for num in draw]
    spectrum = np.abs(fft.fft(flat_past_draws))
    frequencies = fft.fftfreq(len(flat_past_draws))

    sorted_indices = np.argsort(spectrum)[::-1]
    top_frequencies = frequencies[sorted_indices[:lottery_type_instance.pieces_of_draw_numbers]]

    predicted_numbers = set()
    for freq in top_frequencies:
        number = int(np.round(freq * (
                    lottery_type_instance.max_number - lottery_type_instance.min_number) + lottery_type_instance.min_number))
        if lottery_type_instance.min_number <= number <= lottery_type_instance.max_number:
            predicted_numbers.add(number)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(np.random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))

    return sorted(list(predicted_numbers))