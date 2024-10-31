# cross_draw_correlation_prediction.py
from collections import Counter
from typing import List, Tuple, Set, Dict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Cross draw correlation predictor for combined lottery types.
    Returns (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool,
        window_size: int = 5
) -> List[int]:
    """Generate a set of numbers using cross-draw correlation analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    # Calculate correlations
    correlations = analyze_cross_correlations(past_draws, window_size)

    # Generate predictions
    predicted_numbers = generate_predictions(
        correlations,
        min_num,
        max_num,
        required_numbers
    )

    # Ensure we have enough numbers
    fill_remaining_numbers(predicted_numbers, min_num, max_num, required_numbers)

    return sorted(list(predicted_numbers))[:required_numbers]


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def analyze_cross_correlations(
        past_draws: List[List[int]],
        window_size: int
) -> Counter:
    """Analyze correlations between numbers across draws."""
    cross_correlation_counter = Counter()

    for i in range(len(past_draws) - 1):
        current_draw = past_draws[i]
        # Compare with next window_size draws
        for j in range(1, min(window_size + 1, len(past_draws) - i)):
            next_draw = past_draws[i + j]
            for number in current_draw:
                for next_number in next_draw:
                    cross_correlation_counter[(number, next_number)] += 1

    return cross_correlation_counter


def generate_predictions(
        correlations: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> Set[int]:
    """Generate predictions based on correlations."""
    predicted_numbers = set()

    # Get top correlations
    top_correlations = correlations.most_common(required_numbers * 2)

    # Extract numbers from correlations
    for (num1, num2), _ in top_correlations:
        if min_num <= num1 <= max_num:
            predicted_numbers.add(num1)
        if min_num <= num2 <= max_num:
            predicted_numbers.add(num2)
        if len(predicted_numbers) >= required_numbers:
            break

    return predicted_numbers


def fill_remaining_numbers(
        numbers: Set[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> None:
    """Fill remaining slots with random numbers."""
    while len(numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        numbers.add(new_number)


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))


def analyze_correlation_statistics(
        past_draws: List[List[int]],
        top_n: int = 5,
        window_size: int = 5
) -> Dict[Tuple[int, int], int]:
    """
    Analyze and return correlation statistics.

    Parameters:
    - past_draws: Historical draw data
    - top_n: Number of top correlations to return
    - window_size: Number of draws to look ahead

    Returns:
    - Dictionary of top correlations and their counts
    """
    correlations = analyze_cross_correlations(past_draws, window_size)
    return dict(correlations.most_common(top_n))