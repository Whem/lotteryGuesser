import random
import numpy as np
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on quadratic pattern prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)
    past_draws = list(past_draws_queryset)

    # Flatten the list of past numbers and remove duplicates
    all_past_numbers = []
    for draw in past_draws:
        if isinstance(draw, list):
            all_past_numbers.extend(draw)
    all_past_numbers = sorted(set(all_past_numbers))

    if len(all_past_numbers) >= 3:
        # Fit a quadratic polynomial to the past numbers
        x = np.arange(len(all_past_numbers))
        y = np.array(all_past_numbers)
        coefficients = np.polyfit(x, y, 2)
        quadratic = np.poly1d(coefficients)
        # Predict the next numbers based on the quadratic function
        next_indices = np.arange(len(all_past_numbers), len(all_past_numbers) + total_numbers * 2)
        predicted_numbers = quadratic(next_indices)
        # Round and convert to integers
        predicted_numbers = [int(round(num)) for num in predicted_numbers]
        # Filter numbers within the valid range and remove duplicates
        predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]
        predicted_numbers = list(set(predicted_numbers))
    else:
        predicted_numbers = []

    # Ensure we have enough numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        used_numbers = set(predicted_numbers)
        remaining_numbers = list(all_numbers - used_numbers)
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    elif len(predicted_numbers) > total_numbers:
        predicted_numbers = predicted_numbers[:total_numbers]

    # Sort and return the selected numbers
    predicted_numbers.sort()
    return predicted_numbers
