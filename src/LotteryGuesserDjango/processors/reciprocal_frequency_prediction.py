import random
from collections import Counter
from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on repeating pattern prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_draws = [
        tuple(sorted(draw)) for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == lottery_type_instance.pieces_of_draw_numbers
    ]

    # Count frequency of each pattern
    pattern_counter = Counter(past_draws)

    # Find repeating patterns
    repeating_patterns = [pattern for pattern, count in pattern_counter.items() if count > 1]

    if repeating_patterns:
        # Choose the most frequent repeating pattern
        most_common_pattern = max(repeating_patterns, key=lambda p: pattern_counter[p])
        selected_numbers = list(most_common_pattern)
    else:
        # If no repeating patterns, select random numbers
        min_num = lottery_type_instance.min_number
        max_num = lottery_type_instance.max_number
        total_numbers = lottery_type_instance.pieces_of_draw_numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)

    # Ensure unique numbers and correct count
    selected_numbers = list(set(selected_numbers))
    total_needed = lottery_type_instance.pieces_of_draw_numbers
    if len(selected_numbers) < total_needed:
        min_num = lottery_type_instance.min_number
        max_num = lottery_type_instance.max_number
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_needed - len(selected_numbers)])
    elif len(selected_numbers) > total_needed:
        selected_numbers = selected_numbers[:total_needed]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
