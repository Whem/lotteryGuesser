import random
from collections import Counter
from algorithms.models import lg_lottery_winner_number
import datetime

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on time-based number trend prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Define the number of recent weeks to analyze
    recent_weeks = 10  # Adjust as needed

    # Get current year and week
    current_date = datetime.date.today()
    current_year, current_week, _ = current_date.isocalendar()

    # Initialize counters for recent and previous periods
    recent_numbers = Counter()
    previous_numbers = Counter()

    # Collect numbers from recent weeks
    for week_offset in range(recent_weeks):
        # Calculate target week and year
        target_week = current_week - week_offset
        target_year = current_year

        # Adjust for year change if necessary
        while target_week < 1:
            target_year -= 1
            last_week_of_year = datetime.date(target_year, 12, 28).isocalendar()[1]
            target_week += last_week_of_year

        # Retrieve draws for the target week and year
        draws = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance,
            lottery_type_number_year=target_year,
            lottery_type_number_week=target_week
        ).values_list('lottery_type_number', flat=True)

        for draw in draws:
            if isinstance(draw, list):
                for num in draw:
                    if isinstance(num, int):
                        recent_numbers[num] += 1

    # Collect numbers from previous weeks (same number of weeks)
    for week_offset in range(recent_weeks, recent_weeks * 2):
        # Calculate target week and year
        target_week = current_week - week_offset
        target_year = current_year

        # Adjust for year change if necessary
        while target_week < 1:
            target_year -= 1
            last_week_of_year = datetime.date(target_year, 12, 28).isocalendar()[1]
            target_week += last_week_of_year

        # Retrieve draws for the target week and year
        draws = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance,
            lottery_type_number_year=target_year,
            lottery_type_number_week=target_week
        ).values_list('lottery_type_number', flat=True)

        for draw in draws:
            if isinstance(draw, list):
                for num in draw:
                    if isinstance(num, int):
                        previous_numbers[num] += 1

    # Calculate trend scores for each number
    trend_scores = {}
    all_numbers = set(recent_numbers.keys()).union(previous_numbers.keys())
    for num in all_numbers:
        recent_freq = recent_numbers.get(num, 0)
        previous_freq = previous_numbers.get(num, 0)
        if previous_freq == 0:
            trend = recent_freq
        else:
            trend = recent_freq / previous_freq
        trend_scores[num] = trend

    # Sort numbers based on trend scores in descending order
    sorted_numbers = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the top numbers based on the required pieces_of_draw_numbers
    num_to_select = lottery_type_instance.pieces_of_draw_numbers
    selected_numbers = [num for num, trend in sorted_numbers if trend > 1][:num_to_select]

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
