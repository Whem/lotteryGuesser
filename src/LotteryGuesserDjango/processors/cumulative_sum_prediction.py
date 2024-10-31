# cumulative_sum_prediction.py
import random
from collections import Counter
from typing import List, Set, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Cumulative sum predictor for combined lottery types.
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
    """Generate a set of numbers using cumulative sum analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    # Calculate cumulative sums
    sums = analyze_cumulative_sums(past_draws)

    # Try to generate numbers with common sums
    for target_sum, _ in sums.most_common(3):
        predicted_numbers = find_numbers_with_sum(
            target_sum,
            min_num,
            max_num,
            required_numbers
        )
        if predicted_numbers:
            return sorted(predicted_numbers)

    # Fallback to random numbers if no valid combinations found
    return random_number_set(min_num, max_num, required_numbers)


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


def analyze_cumulative_sums(past_draws: List[List[int]]) -> Counter:
    """Analyze cumulative sums from past draws."""
    sums = [sum(draw) for draw in past_draws if draw]
    return Counter(sums)


def find_numbers_with_sum(
        target_sum: int,
        min_num: int,
        max_num: int,
        required_numbers: int,
        max_attempts: int = 1000
) -> List[int]:
    """Find a set of numbers that sum to target_sum."""
    for _ in range(max_attempts):
        numbers = set()
        current_sum = 0

        while current_sum < target_sum and len(numbers) < required_numbers:
            remaining = target_sum - current_sum
            max_possible = min(remaining, max_num)

            if max_possible < min_num:
                break

            new_number = random.randint(min_num, max_possible)
            if new_number not in numbers:
                numbers.add(new_number)
                current_sum += new_number

        if (current_sum == target_sum and
                len(numbers) == required_numbers and
                all(min_num <= n <= max_num for n in numbers)):
            return list(numbers)

    return []


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))


def analyze_sum_statistics(past_draws: List[List[int]]) -> Dict[str, float]:
    """
    Analyze cumulative sum statistics.

    Returns a dictionary containing:
    - min_sum
    - max_sum
    - mean_sum
    - median_sum
    - most_common_sums
    """
    if not past_draws:
        return {}

    sums = [sum(draw) for draw in past_draws if draw]
    if not sums:
        return {}

    stats = {
        'min_sum': min(sums),
        'max_sum': max(sums),
        'mean_sum': sum(sums) / len(sums),
        'median_sum': sorted(sums)[len(sums) // 2]
    }

    # Add most common sums
    sum_counter = Counter(sums)
    stats['most_common_sums'] = dict(sum_counter.most_common(5))

    return stats