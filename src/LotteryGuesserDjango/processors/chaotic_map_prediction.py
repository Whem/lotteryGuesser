# chaotic_map_prediction.py

import numpy as np
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using Deterministic Chaos-Based analysis, supporting both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Main numbers configuration
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Get past draws for main numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id')

    past_main_numbers = [
        draw.lottery_type_number for draw in past_draws_queryset
        if isinstance(draw.lottery_type_number, list)
    ]

    main_numbers = generate_numbers_with_chaos(
        past_numbers=past_main_numbers,
        min_num=min_num,
        max_num=max_num,
        total_numbers=total_numbers
    )

    # Additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        past_additional_numbers = [
            draw.additional_numbers for draw in past_draws_queryset
            if isinstance(draw.additional_numbers, list)
        ]

        additional_numbers = generate_numbers_with_chaos(
            past_numbers=past_additional_numbers,
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count
        )

    return main_numbers, additional_numbers


def generate_numbers_with_chaos(
        past_numbers: List[List[int]],
        min_num: int,
        max_num: int,
        total_numbers: int
) -> List[int]:
    """
    Generates lottery numbers using chaotic map analysis for the given range and count.
    """
    # Fallback to sequential numbers if insufficient data
    if len(past_numbers) < 20:
        return list(range(min_num, min_num + total_numbers))

    # Calculate number frequencies in past draws
    all_numbers = [num for draw in past_numbers for num in draw]
    number_counts = Counter(all_numbers)
    most_common_numbers = [num for num, _ in number_counts.most_common()]

    # Chaotic Map Configuration
    r = 3.9  # Chaos parameter
    iterations = 1000  # Total iterations of the chaotic map
    window_size = 100  # Window size for averaging in predictions

    # Initialize starting point using the average of the last draw
    last_draw = past_numbers[-1]
    x0 = np.mean(last_draw) / (max_num + 1)

    # Generate chaotic sequence
    x = x0
    sequence = []
    for _ in range(iterations):
        x = r * x * (1 - x)
        sequence.append(x)

    # Use the final window of the chaotic sequence for predictions
    recent_sequence = sequence[-window_size:]
    predictions = set()

    # Generate unique predictions
    for i in range(total_numbers):
        avg_val = np.mean(recent_sequence[i:i + window_size // total_numbers])
        predicted_num = int(round(avg_val * (max_num - min_num))) + min_num
        predicted_num = max(min_num, min(max_num, predicted_num))

        # Avoid duplicates by adjusting slightly if necessary
        while predicted_num in predictions:
            predicted_num = (predicted_num % max_num) + min_num

        predictions.add(predicted_num)

    # Fill remaining slots with most common numbers if needed
    while len(predictions) < total_numbers:
        for num in most_common_numbers:
            if num not in predictions:
                predictions.add(num)
                if len(predictions) == total_numbers:
                    break

    return sorted(predictions)

