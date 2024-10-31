# monte_carlo_simulation_prediction.py

import numpy as np
import random
from typing import List, Tuple
from collections import defaultdict

from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using Monte Carlo simulation for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        num_picks=int(lottery_type_instance.pieces_of_draw_numbers),
        iterations=1000
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            num_picks=int(lottery_type_instance.additional_numbers_count),
            iterations=1000
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_numbers(min_num: int, max_num: int, num_picks: int, iterations: int = 1000) -> List[int]:
    """
    Generate numbers using Monte Carlo simulation.
    """
    frequency = monte_carlo_simulation(min_num, max_num, num_picks, iterations)

    most_common = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    predicted_numbers = set()
    for numbers, _ in most_common:
        predicted_numbers.update(numbers)
        if len(predicted_numbers) >= num_picks:
            break

    # Fill missing numbers if not enough
    all_possible_numbers = set(range(min_num, max_num + 1))
    while len(predicted_numbers) < num_picks:
        remaining_numbers = all_possible_numbers - predicted_numbers
        if not remaining_numbers:
            break
        new_number = random.choice(list(remaining_numbers))
        predicted_numbers.add(new_number)

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    return sorted(predicted_numbers)[:num_picks]


def monte_carlo_simulation(min_num: int, max_num: int, num_picks: int, iterations: int) -> defaultdict:
    """
    Perform Monte Carlo simulation to generate possible number combinations.
    """
    frequency = defaultdict(int)
    all_numbers = range(min_num, max_num + 1)
    for _ in range(iterations):
        numbers = tuple(sorted(random.sample(all_numbers, num_picks)))
        frequency[numbers] += 1
    return frequency
