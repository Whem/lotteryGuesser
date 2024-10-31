# quadratic_residue_prediction.py
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on quadratic residue prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return main_numbers, additional_numbers

def generate_number_set(
        lottery_type_instance,
        min_num,
        max_num,
        total_numbers,
        is_main
):
    """
    Generates a set of lottery numbers based on quadratic residue prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.
    - is_main: Boolean indicating whether this is for main numbers or additional numbers.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    modulus = max_num + 1

    # Generate quadratic residues within the lottery number range
    quadratic_residues = set()
    for n in range(modulus):
        residue = (n * n) % modulus
        if min_num <= residue <= max_num:
            quadratic_residues.add(residue)

    # Retrieve past winning numbers
    if is_main:
        past_draws = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).values_list('lottery_type_number', flat=True)
    else:
        past_draws = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).values_list('additional_numbers', flat=True)

    # Count frequency of quadratic residues in past winning numbers
    residue_counter = {}
    for draw in past_draws:
        if isinstance(draw, list):
            for number in draw:
                if number in quadratic_residues:
                    residue_counter[number] = residue_counter.get(number, 0) + 1

    # Sort residues by their frequency in descending order
    sorted_residues = sorted(residue_counter.items(), key=lambda x: x[1], reverse=True)
    selected_numbers = [num for num, _ in sorted_residues]

    # If not enough numbers, fill with random quadratic residues
    if len(selected_numbers) < total_numbers:
        remaining_residues = list(quadratic_residues - set(selected_numbers))
        random.shuffle(remaining_residues)
        selected_numbers.extend(remaining_residues[:total_numbers - len(selected_numbers)])

    # If still not enough, fill with random numbers from the range
    if len(selected_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Ensure we have the correct number of numbers
    selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
