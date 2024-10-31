# recurrence_quantification_analysis_prediction.py

import numpy as np
from pyunicorn.timeseries import RecurrencePlot
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Recurrence Quantification Analysis (RQA).

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
        number_field='lottery_type_number',
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count)
        )

    return main_numbers, additional_numbers


def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a set of lottery numbers using Recurrence Quantification Analysis (RQA).

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(number_field, flat=True)

    past_draws = [
        [int(num) for num in draw] for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 20:
        # If not enough data, return the smallest 'total_numbers' numbers
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    # Transform past draws into a time series
    draw_matrix = np.array(past_draws)
    time_series = draw_matrix.flatten()

    # Perform Recurrence Quantification Analysis
    # Create a recurrence plot
    rp = RecurrencePlot(
        time_series,
        dim=1,
        tau=1,
        metric='euclidean',
        normalize=False,
        recurrence_rate=0.1  # Set a recurrence rate
    )

    # Get the recurrence matrix
    recurrence_matrix = rp.recurrence_matrix()

    # Compute the recurrence frequency for each time point
    recurrence_histogram = np.sum(recurrence_matrix, axis=0)

    # Map recurrence frequency to numbers
    number_scores = {}
    for idx, count in enumerate(recurrence_histogram):
        number = int(time_series[idx])
        if min_num <= number <= max_num:
            number_scores[number] = number_scores.get(number, 0) + count

    # Sort numbers by their recurrence frequency
    sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the first 'total_numbers' numbers
    predicted_numbers = [num for num, score in sorted_numbers]

    # Remove duplicates and keep only valid numbers
    predicted_numbers = [
        int(num) for num in dict.fromkeys(predicted_numbers)
        if min_num <= num <= max_num
    ]

    # If we have fewer numbers than needed, fill with the most frequent numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = time_series
        number_counts = {}
        for number in all_numbers:
            number = int(number)
            if min_num <= number <= max_num:
                number_counts[number] = number_counts.get(number, 0) + 1
        sorted_counts = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        for num, _ in sorted_counts:
            num = int(num)
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Ensure we have exactly 'total_numbers' numbers
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
