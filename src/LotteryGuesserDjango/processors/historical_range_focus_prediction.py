# historical_range_focus_prediction.py
import random
from collections import Counter
from typing import List, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Range-focused predictor for combined lottery types."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

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
    """Generate numbers using range analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)
    range_counter = analyze_historical_ranges(past_draws)

    predicted_numbers = generate_predictions(
        range_counter,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
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


def analyze_historical_ranges(past_draws: List[List[int]]) -> Counter:
    """Analyze historical number ranges."""
    range_counter = Counter()
    for draw in past_draws:
        sorted_draw = sorted(draw)
        range_min, range_max = sorted_draw[0], sorted_draw[-1]
        range_counter[(range_min, range_max)] += 1
    return range_counter


def generate_predictions(
        range_counter: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions from range analysis."""
    predicted_numbers = set()
    top_ranges = range_counter.most_common(3)

    for range_tuple, _ in top_ranges:
        range_min, range_max = range_tuple
        numbers_to_generate = max(1, required_numbers // 3)
        predicted_numbers.update(
            generate_numbers_in_range(range_min, range_max, numbers_to_generate)
        )

    # Fill remaining numbers
    while len(predicted_numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        predicted_numbers.add(new_number)

    return list(predicted_numbers)[:required_numbers]


def generate_numbers_in_range(range_min: int, range_max: int, count: int) -> List[int]:
    """Generate random numbers within range."""
    return random.sample(
        range(range_min, range_max + 1),
        min(count, range_max - range_min + 1)
    )


def calculate_range_statistics(range_counter: Counter) -> Dict[str, float]:
    """Calculate range statistics."""
    total_draws = sum(range_counter.values())
    total_range = sum(
        (max_val - min_val)
        for (min_val, max_val) in range_counter.keys()
    )

    range_sizes = [
        max_val - min_val
        for (min_val, max_val) in range_counter.keys()
    ]

    return {
        'avg_range': total_range / total_draws,
        'median_range': sorted(range_sizes)[len(range_sizes) // 2]
    }