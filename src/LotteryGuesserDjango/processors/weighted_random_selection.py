import random
from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using weighted random selection based on past frequencies.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Count frequency of each number
    number_counter = Counter()
    for draw in past_draws:
        if isinstance(draw, list):
            for number in draw:
                if isinstance(number, int):
                    number_counter[number] += 1

    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    all_numbers = list(range(min_num, max_num + 1))

    # Calculate weights based on frequencies
    frequencies = []
    for number in all_numbers:
        freq = number_counter.get(number, 0)
        frequencies.append(freq)

    # If all frequencies are zero (no past data), assign equal weights
    if all(f == 0 for f in frequencies):
        weights = [1] * len(all_numbers)
    else:
        # Normalize frequencies to get weights
        total_freq = sum(frequencies)
        weights = [freq / total_freq for freq in frequencies]

    # Select numbers based on weights without replacement
    selected_numbers = []
    available_numbers = all_numbers.copy()
    available_weights = weights.copy()

    for _ in range(total_numbers):
        if not available_numbers:
            break
        selected_number = random.choices(available_numbers, weights=available_weights, k=1)[0]
        selected_numbers.append(selected_number)
        # Remove selected number and its weight
        index = available_numbers.index(selected_number)
        del available_numbers[index]
        del available_weights[index]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
