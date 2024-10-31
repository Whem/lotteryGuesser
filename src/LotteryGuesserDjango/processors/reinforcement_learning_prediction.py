# reinforcement_learning_prediction.py

import random
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a simple reinforcement learning approach.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance=lottery_type_instance,
        number_field='lottery_type_number',
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count
        )

    return main_numbers, additional_numbers


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using a simple reinforcement learning approach.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Initialize Q-values for each number
    q_values = {num: 0 for num in range(min_num, max_num + 1)}

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(number_field, flat=True)

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
    # Generate more numbers to increase the chance of uniqueness
    generated_numbers = random.choices(
        population=list(range(min_num, max_num + 1)),
        weights=probabilities,
        k=total_numbers * 2
    )

    # Ensure unique numbers
    selected_numbers = list(set(generated_numbers))

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < total_numbers:
        remaining_numbers = list(set(range(min_num, max_num + 1)) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Ensure we have exactly total_numbers numbers
    selected_numbers = selected_numbers[:total_numbers]
    selected_numbers.sort()
    return selected_numbers
