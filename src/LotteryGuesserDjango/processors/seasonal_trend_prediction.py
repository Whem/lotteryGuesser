import random
from collections import Counter
from algorithms.models import lg_lottery_winner_number
import datetime

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on seasonal trend prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Get current week number
    today = datetime.date.today()
    current_week = today.isocalendar()[1]

    # Define seasons based on week numbers
    if 1 <= current_week <= 13:
        season_weeks = list(range(1, 14))  # Winter
    elif 14 <= current_week <= 26:
        season_weeks = list(range(14, 27))  # Spring
    elif 27 <= current_week <= 39:
        season_weeks = list(range(27, 40))  # Summer
    else:
        season_weeks = list(range(40, 53))  # Autumn

    # Retrieve past winning numbers during the same season
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance,
        lottery_type_number_week__in=season_weeks
    ).values_list('lottery_type_number', flat=True)

    # Count frequency of each number during the current season
    number_counter = Counter()
    for draw in past_draws:
        if isinstance(draw, list):
            for number in draw:
                if isinstance(number, int):
                    number_counter[number] += 1

    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Select numbers based on frequency
    if number_counter:
        most_common_numbers = [num for num, _ in number_counter.most_common()]
        selected_numbers = most_common_numbers[:total_numbers]
    else:
        selected_numbers = []

    # Fill remaining slots with random numbers if needed
    if len(selected_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        used_numbers = set(selected_numbers)
        remaining_numbers = list(all_numbers - used_numbers)
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Ensure correct number of numbers
    selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the numbers
    selected_numbers.sort()
    return selected_numbers
