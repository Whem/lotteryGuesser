# quantum_inspired_optimization_prediction.py

import numpy as np
import random
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def quantum_inspired_random(min_val, max_val, n, prob_dist):
    """
    Generate quantum-inspired random numbers.
    """
    range_size = max_val - min_val + 1
    phase = np.random.random(range_size) * 2 * np.pi
    amplitude = np.sqrt(prob_dist)
    quantum_state = amplitude * np.exp(1j * phase)
    prob = np.abs(quantum_state) ** 2
    prob /= np.sum(prob)  # Normalize probabilities
    return np.random.choice(range(min_val, max_val + 1), size=n, p=prob)

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using quantum-inspired optimization.

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
    Generates a set of lottery numbers using quantum-inspired optimization.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.
    - is_main: Boolean indicating whether this is for main numbers or additional numbers.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    if is_main:
        past_draws = list(
            lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id')[:50].values_list('lottery_type_number', flat=True)
        )
    else:
        past_draws = list(
            lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id')[:50].values_list('additional_numbers', flat=True)
        )

    if len(past_draws) < 10:
        # If not enough past draws, return random numbers
        return sorted(random.sample(range(min_num, max_num + 1), total_numbers))

    # Calculate the frequency of each number
    number_frequency = {i: 1 for i in range(min_num, max_num + 1)}  # Initialize with 1 to avoid zero probabilities
    for draw in past_draws:
        if isinstance(draw, list):
            for num in draw:
                if min_num <= num <= max_num:
                    number_frequency[num] += 1

    # Create probability distribution
    prob_distribution = [number_frequency[i] for i in range(min_num, max_num + 1)]
    prob_distribution = np.array(prob_distribution, dtype=float)
    prob_distribution /= np.sum(prob_distribution)  # Normalize

    # Generate quantum-inspired numbers based on the probability distribution
    quantum_numbers = quantum_inspired_random(min_num, max_num, total_numbers * 2, prob_distribution)

    # Select unique numbers
    selected_numbers = []
    for num in quantum_numbers:
        if num not in selected_numbers:
            selected_numbers.append(int(num))  # Convert to Python int
        if len(selected_numbers) == total_numbers:
            break

    # If not enough numbers, fill with random selection
    if len(selected_numbers) < total_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(selected_numbers)
        additional_numbers = random.sample(remaining, total_numbers - len(selected_numbers))
        selected_numbers += additional_numbers

    return sorted(selected_numbers)
