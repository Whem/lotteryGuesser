# monte_carlo_simulation_prediction.py

import numpy as np
import random
from typing import List
from collections import defaultdict

from algorithms.models import lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    def monte_carlo_simulation(min_num, max_num, iterations=1000):
        frequency = defaultdict(int)
        for _ in range(iterations):
            numbers = tuple(sorted(random.sample(range(min_num, max_num + 1), lottery_type_instance.pieces_of_draw_numbers)))
            frequency[numbers] += 1
        return frequency

    frequency = monte_carlo_simulation(lottery_type_instance.min_number, lottery_type_instance.max_number)

    most_common = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    predicted_numbers = set()
    for numbers, _ in most_common:
        predicted_numbers.update(numbers)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        predicted_numbers.add(new_number)

    return sorted([int(num) for num in predicted_numbers])[:lottery_type_instance.pieces_of_draw_numbers]
