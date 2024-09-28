import random
from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on odd-even balance prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Analyze odd-even patterns
    pattern_counter = Counter()
    for draw in past_draws:
        if not isinstance(draw, list):
            continue
        odd_count = sum(1 for number in draw if number % 2 != 0)
        even_count = sum(1 for number in draw if number % 2 == 0)
        pattern_counter[(odd_count, even_count)] += 1

    # Find the most common odd-even pattern
    most_common_pattern = pattern_counter.most_common(1)
    if most_common_pattern:
        odd_count, even_count = most_common_pattern[0][0]
    else:
        # Default to an even split if no data is available
        total_numbers = lottery_type_instance.pieces_of_draw_numbers
        odd_count = total_numbers // 2
        even_count = total_numbers - odd_count

    # Generate numbers matching the most common pattern
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    all_numbers = list(range(min_num, max_num + 1))

    odd_numbers = [num for num in all_numbers if num % 2 != 0]
    even_numbers = [num for num in all_numbers if num % 2 == 0]

    # Check if there are enough odd and even numbers
    odd_count = min(odd_count, len(odd_numbers))
    even_count = min(even_count, len(even_numbers))

    # Randomly select the required number of odd and even numbers
    selected_numbers = []
    selected_numbers.extend(random.sample(odd_numbers, odd_count))
    selected_numbers.extend(random.sample(even_numbers, even_count))

    # Ensure we have the correct number of total numbers
    num_to_select = lottery_type_instance.pieces_of_draw_numbers
    if len(selected_numbers) < num_to_select:
        # Fill with random numbers from the remaining pool
        remaining_numbers = list(set(all_numbers) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:num_to_select - len(selected_numbers)])
    elif len(selected_numbers) > num_to_select:
        # Trim the list if we have too many numbers
        random.shuffle(selected_numbers)
        selected_numbers = selected_numbers[:num_to_select]

    # Sort the final list of numbers
    selected_numbers.sort()
    return selected_numbers
