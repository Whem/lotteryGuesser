# quantum_rng_simulation.py

import numpy as np
from typing import List, Tuple
from algorithms.models import lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a quantum RNG simulation.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count
        )

    return main_numbers, additional_numbers


def generate_number_set(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generates a set of lottery numbers using quantum RNG simulation.

    Parameters:
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    def quantum_measurement():
        # Simulated quantum measurement
        return np.random.choice([0, 1], p=[0.5, 0.5])

    def generate_quantum_number(min_num, max_num):
        range_size = max_num - min_num + 1
        bits_needed = int(np.ceil(np.log2(range_size)))

        while True:
            quantum_bits = [quantum_measurement() for _ in range(bits_needed)]
            number = sum(bit << i for i, bit in enumerate(quantum_bits))
            if number < range_size:
                return int(min_num + number)

    predicted_numbers = set()
    while len(predicted_numbers) < total_numbers:
        num = generate_quantum_number(min_num, max_num)
        predicted_numbers.add(num)

    return sorted(predicted_numbers)[:total_numbers]
