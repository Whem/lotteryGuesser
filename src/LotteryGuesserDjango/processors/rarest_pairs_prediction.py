# rarest_pairs_prediction.py
import random
from collections import Counter
from itertools import combinations
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on the rarest pairs prediction.

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
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers,
        number_field='lottery_type_number'
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance=lottery_type_instance,
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count,
            number_field='additional_numbers'
        )

    return main_numbers, additional_numbers

def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_numbers: int,
    number_field: str
) -> List[int]:
    """
    Generates a set of lottery numbers based on the rarest pairs prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers ('lottery_type_number' or 'additional_numbers').

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(number_field, flat=True)

    # Count frequency of each pair
    pair_counter = Counter()
    for draw in past_draws:
        if not isinstance(draw, list):
            continue
        # Generate all possible pairs from the draw
        draw_pairs = combinations(sorted(set(draw)), 2)
        for pair in draw_pairs:
            pair_counter[pair] += 1

    # Find the least common pairs
    if pair_counter:
        # Get the maximum frequency to find the least frequent pairs
        max_frequency = max(pair_counter.values())
        # Invert the frequencies to sort by least frequent
        inverted_pair_counter = {pair: max_frequency - count for pair, count in pair_counter.items()}
        # Sort pairs by their inverted frequency in descending order (rarest first)
        rarest_pairs = sorted(inverted_pair_counter.items(), key=lambda x: x[1], reverse=True)
    else:
        rarest_pairs = []

    # Build a set of numbers from the rarest pairs
    selected_numbers = set()

    # Iterate over the rarest pairs and add numbers to the selection
    for pair, _ in rarest_pairs:
        selected_numbers.update(pair)
        if len(selected_numbers) >= total_numbers:
            break

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - selected_numbers)
        random.shuffle(remaining_numbers)
        selected_numbers.update(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Convert to a sorted list
    selected_numbers = sorted(selected_numbers)[:total_numbers]

    return selected_numbers
