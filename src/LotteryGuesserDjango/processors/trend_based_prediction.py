import random
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on trend-based prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Define the number of recent draws to analyze
    recent_draws_count = 20  # Adjust as needed

    # Retrieve recent draws
    recent_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:recent_draws_count].values_list('lottery_type_number', flat=True)

    # Count frequency of each number in recent draws
    recent_frequency = defaultdict(int)
    for draw in recent_draws:
        if isinstance(draw, list):
            for num in draw:
                if isinstance(num, int):
                    recent_frequency[num] += 1

    # Retrieve previous draws
    previous_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[recent_draws_count:recent_draws_count*2].values_list('lottery_type_number', flat=True)

    # Count frequency of each number in previous draws
    previous_frequency = defaultdict(int)
    for draw in previous_draws:
        if isinstance(draw, list):
            for num in draw:
                if isinstance(num, int):
                    previous_frequency[num] += 1

    # Calculate trend score for each number
    trend_scores = {}
    all_numbers = set(recent_frequency.keys()).union(previous_frequency.keys())
    for num in all_numbers:
        recent_freq = recent_frequency.get(num, 0)
        prev_freq = previous_frequency.get(num, 0)
        # Trend score is the difference in frequencies
        trend_scores[num] = recent_freq - prev_freq

    # Sort numbers by trend scores in descending order
    sorted_numbers = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)

    # Select numbers with positive trend scores
    num_to_select = lottery_type_instance.pieces_of_draw_numbers
    selected_numbers = [num for num, score in sorted_numbers if score > 0][:num_to_select]

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < num_to_select:
        min_num = lottery_type_instance.min_number
        max_num = lottery_type_instance.max_number
        remaining_numbers = list(set(range(min_num, max_num + 1)) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:num_to_select - len(selected_numbers)])

    # Ensure the correct number of numbers
    selected_numbers = selected_numbers[:num_to_select]

    # Sort and return the numbers
    selected_numbers.sort()
    return selected_numbers
