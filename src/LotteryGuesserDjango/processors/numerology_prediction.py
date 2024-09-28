import datetime
import random
from algorithms.models import lg_lottery_winner_number

def get_life_path_number(date_of_birth):
    """
    Calculates the life path number based on a given date of birth.
    """
    total = sum(int(digit) for digit in date_of_birth.replace('-', '') if digit.isdigit())
    while total > 9:
        total = sum(int(digit) for digit in str(total))
    return total

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on numerology prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Example date of birth (can be modified or taken as input)
    date_of_birth = '1990-01-01'  # YYYY-MM-DD format

    # Calculate life path number
    life_path_number = get_life_path_number(date_of_birth)

    # Use life path number to generate base numbers
    base_numbers = []
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number

    # Generate numbers based on numerology calculations
    for i in range(1, lottery_type_instance.pieces_of_draw_numbers + 1):
        num = (life_path_number * i) % (max_num - min_num + 1) + min_num
        base_numbers.append(num)

    # Ensure numbers are unique
    selected_numbers = list(set(base_numbers))

    # If not enough unique numbers, fill with random numbers
    num_to_select = lottery_type_instance.pieces_of_draw_numbers
    if len(selected_numbers) < num_to_select:
        all_numbers = set(range(min_num, max_num + 1))
        used_numbers = set(selected_numbers)
        remaining_numbers = list(all_numbers - used_numbers)
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:num_to_select - len(selected_numbers)])

    # Sort the final list of numbers
    selected_numbers = sorted(selected_numbers)[:num_to_select]
    return selected_numbers
