#chaos theory prediction
import numpy as np
import random
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate main and additional numbers using chaos theory (logistic map) for prediction.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Fallback to random selection if insufficient past data
    if len(past_draws) < 50:
        main_numbers = random.sample(
            range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
            lottery_type_instance.pieces_of_draw_numbers
        )
        additional_numbers = random.sample(
            range(lottery_type_instance.additional_min_number, lottery_type_instance.additional_max_number + 1),
            lottery_type_instance.additional_numbers_count
        ) if lottery_type_instance.has_additional_numbers else []

        return sorted(main_numbers), sorted(additional_numbers)

    # Generate chaotic sequence with logistic map for main numbers
    main_numbers = generate_chaotic_numbers(
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers
    )

    # Generate chaotic sequence with logistic map for additional numbers if required
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_chaotic_numbers(
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_chaotic_numbers(min_num: int, max_num: int, count: int) -> List[int]:
    """
    Generate a list of numbers using a chaotic sequence scaled to a specified range.
    """
    r = 3.99  # Chaos parameter for the logistic map
    x = 0.5  # Initial value for chaos sequence

    chaotic_sequence = []
    for _ in range(1000):
        x = r * x * (1 - x)
        chaotic_sequence.append(x)

    # Scale chaotic sequence to the desired range
    scaled_sequence = np.interp(
        chaotic_sequence,
        (min(chaotic_sequence), max(chaotic_sequence)),
        (min_num, max_num)
    )

    # Select unique numbers from the chaotic sequence until reaching the desired count
    predicted_numbers = set()
    for num in scaled_sequence:
        rounded_num = int(round(num))
        if min_num <= rounded_num <= max_num:
            predicted_numbers.add(rounded_num)
        if len(predicted_numbers) == count:
            break

    return list(predicted_numbers)
