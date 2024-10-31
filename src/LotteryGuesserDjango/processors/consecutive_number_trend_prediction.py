# consecutive_number_trend_prediction.py
import random
from collections import Counter
from typing import List, Tuple, Set,Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Consecutive number trend predictor for combined lottery types.
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
        is_main: bool
) -> List[int]:
    """Generate a set of numbers using consecutive trend analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    # Analyze consecutive trends
    consecutive_trends = analyze_consecutive_trends(past_draws)

    # Generate predictions
    predicted_numbers = generate_predictions(
        consecutive_trends,
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


def analyze_consecutive_trends(past_draws: List[List[int]]) -> Counter:
    """Analyze consecutive number trends in past draws."""
    consecutive_trends = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            if sorted_draw[i] + 1 == sorted_draw[i + 1]:
                consecutive_trends[(sorted_draw[i], sorted_draw[i + 1])] += 1

    return consecutive_trends


def generate_predictions(
        consecutive_trends: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> Set[int]:
    """Generate predictions based on consecutive trends."""
    predicted_numbers = set()

    # Use top N most common trends
    N = min(3, required_numbers // 2)
    common_trends = consecutive_trends.most_common(N)

    for trend, _ in common_trends:
        predicted_numbers.update(trend)
        if len(predicted_numbers) >= required_numbers:
            break

        # Extend trends in both directions
        extend_trend(predicted_numbers, trend[0] - 1, min_num, -1, required_numbers)
        if len(predicted_numbers) >= required_numbers:
            break
        extend_trend(predicted_numbers, trend[1] + 1, max_num, 1, required_numbers)
        if len(predicted_numbers) >= required_numbers:
            break

    return predicted_numbers


def extend_trend(
        numbers: Set[int],
        start: int,
        limit: int,
        step: int,
        max_numbers: int
) -> None:
    """Extend a trend in a given direction."""
    current = start
    while len(numbers) < max_numbers and current != limit:
        if limit <= current <= limit:
            numbers.add(current)
        current += step


def fill_remaining_numbers(
        numbers: Set[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> None:
    """Fill remaining slots with random numbers."""
    all_numbers = set(range(min_num, max_num + 1))
    available_numbers = list(all_numbers - numbers)

    while len(numbers) < required_numbers and available_numbers:
        new_number = random.choice(available_numbers)
        numbers.add(new_number)
        available_numbers.remove(new_number)


def analyze_trends(past_draws: List[List[int]]) -> Dict[int, int]:
    """Analyze number difference trends in past draws."""
    all_trends = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            all_trends[sorted_draw[i + 1] - sorted_draw[i]] += 1

    return dict(all_trends.most_common(5))