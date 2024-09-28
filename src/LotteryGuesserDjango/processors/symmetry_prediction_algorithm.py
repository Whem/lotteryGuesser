import random
from algorithms.models import lg_lottery_winner_number

def is_symmetric(number):
    number_str = str(number)
    return number_str == number_str[::-1]

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on the symmetry prediction algorithm.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Generate all symmetrical numbers within the range
    symmetrical_numbers = [num for num in range(min_num, max_num + 1) if is_symmetric(num)]

    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Count frequency of symmetrical numbers in past draws
    symmetry_counter = {}
    for draw in past_draws:
        if isinstance(draw, list):
            for number in draw:
                if is_symmetric(number):
                    symmetry_counter[number] = symmetry_counter.get(number, 0) + 1

    # Sort symmetrical numbers by frequency in descending order
    sorted_symmetrical_numbers = sorted(symmetry_counter.items(), key=lambda x: x[1], reverse=True)
    selected_numbers = [num for num, _ in sorted_symmetrical_numbers]

    # Add remaining symmetrical numbers if needed
    if len(selected_numbers) < total_numbers:
        remaining_symmetrical_numbers = list(set(symmetrical_numbers) - set(selected_numbers))
        random.shuffle(remaining_symmetrical_numbers)
        selected_numbers.extend(remaining_symmetrical_numbers[:total_numbers - len(selected_numbers)])

    # If still not enough, fill with random numbers within the symmetrical range
    if len(selected_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Ensure we have the correct number of numbers
    selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the numbers
    selected_numbers.sort()
    return selected_numbers
