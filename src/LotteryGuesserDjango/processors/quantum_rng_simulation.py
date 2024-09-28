# quantum_rng_simulation.py

import numpy as np
import random
from typing import List
from algorithms.models import lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    def quantum_measurement():
        # Szimulált kvantum mérés
        return np.random.choice([0, 1], p=[0.5, 0.5])

    def generate_quantum_number(min_num, max_num):
        range_size = max_num - min_num + 1
        bits_needed = int(np.ceil(np.log2(range_size)))

        while True:
            quantum_bits = [quantum_measurement() for _ in range(bits_needed)]
            number = sum(bit << i for i, bit in enumerate(quantum_bits))
            if min_num <= number <= max_num:
                return int(number)  # Konvertálás Python int-re

    predicted_numbers = set()
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        num = generate_quantum_number(lottery_type_instance.min_number, lottery_type_instance.max_number)
        predicted_numbers.add(int(num))  # Konvertálás Python int-re

    return sorted([int(num) for num in predicted_numbers])[:lottery_type_instance.pieces_of_draw_numbers]
