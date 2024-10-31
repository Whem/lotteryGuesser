# quantum_entropy_prediction.py

import random
import numpy as np
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a quantum entropy prediction algorithm.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance,
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        total_numbers: int,
        is_main: bool
) -> List[int]:
    """
    Generates a set of lottery numbers using quantum entropy prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.
    - is_main: Boolean indicating whether this is for main numbers or additional numbers.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past draws
    if is_main:
        past_draws = list(
            lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).values_list('lottery_type_number', flat=True)
        )
    else:
        past_draws = list(
            lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).values_list('additional_numbers', flat=True)
        )

    # If not enough past draws, return random numbers
    if len(past_draws) < 20:
        return sorted(random.sample(range(min_num, max_num + 1), total_numbers))

    # Ensure past draws are lists of numbers
    past_draws = [draw for draw in past_draws if isinstance(draw, list) and len(draw) > 0]

    # Simulated quantum entropy generation
    def generate_quantum_entropy():
        return np.random.random()

    # Calculate average entropy from past draws
    entropies = []
    for draw in past_draws:
        if np.mean(draw) > 0:
            entropy = np.std(draw) / np.mean(draw)
            entropies.append(entropy)
    avg_entropy = np.mean(entropies) if entropies else 1

    predicted_numbers = set()
    attempts = 0
    max_attempts = total_numbers * 10  # Prevent infinite loop

    while len(predicted_numbers) < total_numbers and attempts < max_attempts:
        quantum_entropy = generate_quantum_entropy()
        number = int(
            min_num + (quantum_entropy * avg_entropy) * (max_num - min_num)
        )

        if min_num <= number <= max_num:
            predicted_numbers.add(number)

        attempts += 1

    # Fill in if not enough numbers were generated
    if len(predicted_numbers) < total_numbers:
        remaining_numbers = set(range(min_num, max_num + 1)) - predicted_numbers
        predicted_numbers.update(random.sample(
            remaining_numbers,
            total_numbers - len(predicted_numbers)
        ))

    return sorted(predicted_numbers)[:total_numbers]
