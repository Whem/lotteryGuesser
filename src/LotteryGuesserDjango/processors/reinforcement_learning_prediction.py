# reinforcement_learning_prediction.py

import random
import numpy as np
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using a simple reinforcement learning approach.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Note: Reinforcement learning typically requires a defined environment and reward system.
    # For the purpose of this example, we'll use a simplified approach.

    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Initialize Q-values for each number
    q_values = {num: 0 for num in range(min_num, max_num + 1)}

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    # Update Q-values based on past draws
    for draw in past_draws_queryset:
        if isinstance(draw, list):
            for num in draw:
                if min_num <= num <= max_num:
                    q_values[num] += 1  # Simple reward

    # Normalize Q-values to probabilities
    total_q = sum(q_values.values())
    if total_q == 0:
        probabilities = [1 / (max_num - min_num + 1)] * (max_num - min_num + 1)
    else:
        probabilities = [q_values[num] / total_q for num in range(min_num, max_num + 1)]

    # Select numbers based on probabilities
    selected_numbers = random.choices(
        population=range(min_num, max_num + 1),
        weights=probabilities,
        k=total_numbers
    )

    # Ensure unique numbers
    selected_numbers = list(set(selected_numbers))

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < total_numbers:
        remaining_numbers = list(set(range(min_num, max_num + 1)) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    selected_numbers = selected_numbers[:total_numbers]
    selected_numbers.sort()
    return selected_numbers
