import random
from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on reverse order frequency prediction.

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
        if not isinstance(draw, list):
            continue
        for number in draw:
            if isinstance(number, int):
                number_counter[number] += 1

    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    all_numbers = list(range(min_num, max_num + 1))

    # Sort numbers by increasing frequency (least frequent first)
    numbers_by_frequency = sorted(all_numbers, key=lambda x: number_counter.get(x, 0))

    # Select the least frequent numbers
    selected_numbers = numbers_by_frequency[:total_numbers]

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < total_numbers:
        remaining_numbers = list(set(all_numbers) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Ensure we have the correct number of numbers
    selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
