# chaos_theory_prediction.py

import numpy as np
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    if len(past_draws) < 50:
        return random.sample(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
                             lottery_type_instance.pieces_of_draw_numbers)

    # Használjuk a logisztikus leképezést a káosz generálásához
    r = 3.99  # Káosz paraméter
    x = 0.5  # Kezdeti érték

    chaotic_sequence = []
    for _ in range(1000):
        x = r * x * (1 - x)
        chaotic_sequence.append(x)

    # Skálázzuk a kaotikus sorozatot a lottószámok tartományára
    scaled_sequence = np.interp(chaotic_sequence, (min(chaotic_sequence), max(chaotic_sequence)),
                                (lottery_type_instance.min_number, lottery_type_instance.max_number))

    predicted_numbers = set()
    for num in scaled_sequence:
        rounded_num = int(round(num))
        if lottery_type_instance.min_number <= rounded_num <= lottery_type_instance.max_number:
            predicted_numbers.add(rounded_num)
        if len(predicted_numbers) == lottery_type_instance.pieces_of_draw_numbers:
            break

    return sorted(predicted_numbers)