# spectrum_analysis_prediction.py
import random
import numpy as np
from scipy import fft
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Spectrum Analysis Prediction.

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
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count)
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
    Generates a list of lottery numbers using Spectrum Analysis Prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers
                    ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list(number_field, flat=True))

    if len(past_draws) < 20:
        # If not enough data, return random unique numbers
        selected_numbers = sorted(random.sample(range(min_num, max_num + 1), total_numbers))
        return selected_numbers

    # Flatten the past draws into a single list of numbers
    flat_past_draws = [num for draw in past_draws for num in draw if isinstance(num, int)]

    if not flat_past_draws:
        # If no valid past draws, return random unique numbers
        selected_numbers = sorted(random.sample(range(min_num, max_num + 1), total_numbers))
        return selected_numbers

    # Perform Fast Fourier Transform (FFT) on the flattened past draws
    spectrum = np.abs(fft.fft(flat_past_draws))
    frequencies = fft.fftfreq(len(flat_past_draws))

    # Sort the frequencies by their corresponding spectrum magnitude in descending order
    sorted_indices = np.argsort(spectrum)[::-1]
    top_frequencies = frequencies[sorted_indices[:total_numbers * 2]]  # Generate more to increase uniqueness

    predicted_numbers = set()
    for freq in top_frequencies:
        # Map frequency back to a number within the lottery range
        number = int(np.round(freq * (max_num - min_num) + min_num))
        if min_num <= number <= max_num:
            predicted_numbers.add(number)
        if len(predicted_numbers) >= total_numbers:
            break

    # If not enough unique numbers, fill with random numbers
    while len(predicted_numbers) < total_numbers:
        random_num = random.randint(min_num, max_num)
        predicted_numbers.add(random_num)

    # Ensure exactly 'total_numbers' are selected
    selected_numbers = sorted(list(predicted_numbers))[:total_numbers]
    return selected_numbers
