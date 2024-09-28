import random
from collections import Counter
from itertools import combinations
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on pattern matching prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id').values_list('lottery_type_number', flat=True)

    # Analyze patterns in past draws
    pattern_counter = Counter()
    for draw in past_draws:
        if isinstance(draw, list):
            # Create a pattern based on differences between numbers
            sorted_draw = sorted(draw)
            pattern = tuple(sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw)-1))
            pattern_counter[pattern] += 1

    # Find the most common pattern
    most_common_patterns = pattern_counter.most_common()
    if most_common_patterns:
        most_common_pattern = most_common_patterns[0][0]
        # Generate numbers based on the most common pattern
        min_num = lottery_type_instance.min_number
        max_num = lottery_type_instance.max_number
        total_needed = lottery_type_instance.pieces_of_draw_numbers

        # Start with a random starting number within a valid range
        max_start = max_num - sum(most_common_pattern)
        if max_start < min_num:
            max_start = min_num
        start_number = random.randint(min_num, max_start)

        # Build the number sequence based on the pattern
        selected_numbers = [start_number]
        for diff in most_common_pattern:
            next_number = selected_numbers[-1] + diff
            selected_numbers.append(next_number)

        # If generated numbers exceed the valid range, fill with random numbers
        if any(num > max_num or num < min_num for num in selected_numbers) or len(selected_numbers) != total_needed:
            selected_numbers = random.sample(range(min_num, max_num + 1), total_needed)
    else:
        # If no patterns found, select random numbers
        min_num = lottery_type_instance.min_number
        max_num = lottery_type_instance.max_number
        total_needed = lottery_type_instance.pieces_of_draw_numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_needed)

    # Ensure the correct number of unique numbers
    selected_numbers = list(set(selected_numbers))
    if len(selected_numbers) < total_needed:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_needed - len(selected_numbers)])
    elif len(selected_numbers) > total_needed:
        selected_numbers = selected_numbers[:total_needed]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
