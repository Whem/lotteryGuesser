import random
from collections import Counter
from itertools import combinations
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on the rarest pairs prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

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
        rarest_pairs = sorted(inverted_pair_counter.items(), key=lambda x: x[1], reverse=True)
    else:
        rarest_pairs = []

    # Build a set of numbers from the rarest pairs
    selected_numbers = set()
    num_to_select = lottery_type_instance.pieces_of_draw_numbers

    # Iterate over the rarest pairs and add numbers to the selection
    for pair, _ in rarest_pairs:
        selected_numbers.update(pair)
        if len(selected_numbers) >= num_to_select:
            break

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < num_to_select:
        min_num = lottery_type_instance.min_number
        max_num = lottery_type_instance.max_number
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - selected_numbers)
        random.shuffle(remaining_numbers)
        selected_numbers.update(remaining_numbers[:num_to_select - len(selected_numbers)])

    # Convert to a sorted list
    selected_numbers = sorted(selected_numbers)[:num_to_select]

    return selected_numbers
