from collections import defaultdict
import random
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on number wave prediction using existing database fields.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Define the number of past draws to analyze
    draws_to_analyze = 20  # Adjust this value as needed

    # Retrieve the most recent past draws
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:draws_to_analyze].values_list('lottery_type_number', flat=True)

    # Build a list of frequencies in reverse order (most recent first)
    draw_frequencies = []
    for draw in past_draws:
        frequency = defaultdict(int)
        if not isinstance(draw, list):
            continue
        for number in draw:
            if (isinstance(number, int) and
                lottery_type_instance.min_number <= number <= lottery_type_instance.max_number):
                frequency[number] += 1
        draw_frequencies.append(frequency)

    # Calculate wave scores for each number
    wave_scores = defaultdict(float)
    for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
        score = 0
        for i in range(len(draw_frequencies) - 1):
            current_freq = draw_frequencies[i].get(number, 0)
            previous_freq = draw_frequencies[i + 1].get(number, 0)
            # Calculate the difference in frequency
            score += (current_freq - previous_freq)
        wave_scores[number] = score

    # Sort numbers based on wave scores in descending order
    sorted_numbers = sorted(wave_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the top numbers based on the required pieces_of_draw_numbers
    num_to_select = lottery_type_instance.pieces_of_draw_numbers
    selected_numbers = [num for num, score in sorted_numbers if score > 0][:num_to_select]

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < num_to_select:
        all_numbers = set(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))
        used_numbers = set(selected_numbers)
        remaining_numbers = list(all_numbers - used_numbers)
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:num_to_select - len(selected_numbers)])

    # Sort the final list of numbers
    selected_numbers.sort()
    return selected_numbers
