# random_walk_with_momentum_prediction.py
import random
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on random walk with momentum prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance=lottery_type_instance,
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers,
        number_field='lottery_type_number'
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance=lottery_type_instance,
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count,
            number_field='additional_numbers'
        )

    return main_numbers, additional_numbers

def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_numbers: int,
    number_field: str
) -> List[int]:
    """
    Generates a set of lottery numbers based on random walk with momentum prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers ('lottery_type_number' or 'additional_numbers').

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve the most recent winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id').values_list(number_field, flat=True)

    if past_draws.exists():
        # Start from the most recent draw
        last_draw = past_draws.first()
        if isinstance(last_draw, list) and len(last_draw) == total_numbers:
            current_numbers = sorted(last_draw)
        else:
            # If last draw is not usable, start from random numbers
            current_numbers = sorted(random.sample(range(min_num, max_num + 1), total_numbers))
    else:
        # If no past draws, start from random numbers
        current_numbers = sorted(random.sample(range(min_num, max_num + 1), total_numbers))

    # Initialize momentum for each number
    momentums = [0] * total_numbers

    # Perform the random walk with momentum
    predicted_numbers = []
    for idx, num in enumerate(current_numbers):
        # Possible changes: -1, 0, +1
        possible_moves = [-1, 0, 1]

        # Bias towards previous momentum
        if momentums[idx] == 0:
            # No previous momentum, choose randomly
            move = random.choice(possible_moves)
        else:
            # Continue in the same direction with higher probability
            move_options = [momentums[idx]] * 2 + [0, -momentums[idx]]
            move = random.choice(move_options)

        # Update momentum
        momentums[idx] = move

        new_num = num + move

        # Ensure the new number is within bounds
        if new_num < min_num:
            new_num = min_num
        elif new_num > max_num:
            new_num = max_num

        predicted_numbers.append(new_num)

    # Ensure unique numbers
    predicted_numbers = list(set(predicted_numbers))

    # If not enough numbers, fill with random numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    elif len(predicted_numbers) > total_numbers:
        random.shuffle(predicted_numbers)
        predicted_numbers = predicted_numbers[:total_numbers]

    # Sort and return the selected numbers
    predicted_numbers.sort()
    return predicted_numbers
