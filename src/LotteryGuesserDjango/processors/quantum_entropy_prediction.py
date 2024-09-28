# quantum_entropy_prediction.py

import numpy as np
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    if len(past_draws) < 20:
        return random.sample(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
                             lottery_type_instance.pieces_of_draw_numbers)

    # Szimulált kvantum entrópia generálás
    def generate_quantum_entropy():
        return np.random.random()

    # Számítsuk ki az átlagos entrópiát a múltbeli húzásokból
    avg_entropy = np.mean([np.std(draw) / np.mean(draw) for draw in past_draws])

    predicted_numbers = set()
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        quantum_entropy = generate_quantum_entropy()
        number = int(lottery_type_instance.min_number +
                     (quantum_entropy * avg_entropy) * (
                                 lottery_type_instance.max_number - lottery_type_instance.min_number))

        if lottery_type_instance.min_number <= number <= lottery_type_instance.max_number:
            predicted_numbers.add(number)

    return sorted(predicted_numbers)