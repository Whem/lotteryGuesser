from collections import Counter
from typing import List
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type, window_size: int = 5) -> List[int]:
    """
    Generates lottery numbers based on cross-draw correlations with optimized performance.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - window_size: The number of subsequent draws to compare with each draw.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past draws in order
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True))

    cross_correlation_counter = Counter()

    # Calculate cross-draw correlations within the specified window size
    for i in range(len(past_draws) - 1):
        current_draw = past_draws[i]
        # Limit the comparison to the next 'window_size' draws
        for j in range(1, min(window_size + 1, len(past_draws) - i)):
            next_draw = past_draws[i + j]
            for number in current_draw:
                for next_number in next_draw:
                    cross_correlation_counter[(number, next_number)] += 1

    # Select top correlations
    top_correlations = cross_correlation_counter.most_common(
        lottery_type_instance.pieces_of_draw_numbers * 2
    )

    # Extract numbers from top correlations
    predicted_numbers = set()
    for (num1, num2), _ in top_correlations:
        predicted_numbers.add(num1)
        predicted_numbers.add(num2)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    # Fill remaining numbers if needed
    fill_remaining_numbers(predicted_numbers, lottery_type_instance)

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]

def fill_remaining_numbers(numbers: set, lottery_type_instance: lg_lottery_type):
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number
        )
        numbers.add(new_number)

def analyze_correlations(past_draws: List[List[int]], top_n: int = 5, window_size: int = 5) -> None:
    """
    Analyzes cross-draw correlations and prints the top N correlations.

    Parameters:
    - past_draws: A list of past draws, where each draw is a list of numbers.
    - top_n: The number of top correlations to display.
    - window_size: The number of subsequent draws to compare with each draw.
    """
    cross_correlation_counter = Counter()

    for i in range(len(past_draws) - 1):
        current_draw = past_draws[i]
        for j in range(1, min(window_size + 1, len(past_draws) - i)):
            next_draw = past_draws[i + j]
            for number in current_draw:
                for next_number in next_draw:
                    cross_correlation_counter[(number, next_number)] += 1

    print(f"Top {top_n} cross-draw correlations:")
    for (num1, num2), count in cross_correlation_counter.most_common(top_n):
        print(f"Numbers {num1} and {num2} appeared together {count} times within window size {window_size}")
