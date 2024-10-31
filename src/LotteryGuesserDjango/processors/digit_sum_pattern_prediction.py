# digital_sum_pattern_prediction.py
from collections import Counter
from typing import List, Dict, Tuple, Set
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Digital sum pattern predictor for combined lottery types.
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
    """Generate a set of numbers using digital sum patterns."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    # Analyze digit sums
    digit_sum_counter = analyze_digit_sums(past_draws)

    # Get predictions based on common digit sums
    predicted_numbers = generate_predictions(
        digit_sum_counter,
        min_num,
        max_num,
        required_numbers
    )

    # Fill remaining if needed
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


def digit_sum(number: int) -> int:
    """Calculate the sum of digits in a number."""
    return sum(int(digit) for digit in str(number))


def analyze_digit_sums(past_draws: List[List[int]]) -> Counter:
    """Analyze digit sums from past draws."""
    digit_sum_counter = Counter()
    for draw in past_draws:
        for number in draw:
            digit_sum_counter[digit_sum(number)] += 1
    return digit_sum_counter


def generate_predictions(
        digit_sum_counter: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int,
        top_sums: int = 3
) -> Set[int]:
    """Generate predictions based on common digit sums."""
    predicted_numbers = set()

    # Get most common digit sums
    common_sums = [sum for sum, _ in digit_sum_counter.most_common(top_sums)]

    # Create digit sum map for possible numbers
    digit_sum_map = {
        num: digit_sum(num)
        for num in range(min_num, max_num + 1)
    }

    # Generate predictions for each common sum
    for target_sum in common_sums:
        candidates = [
            num for num, sum in digit_sum_map.items()
            if sum == target_sum
        ]
        if candidates:
            sample_size = min(
                required_numbers - len(predicted_numbers),
                len(candidates)
            )
            predicted_numbers.update(
                random.sample(candidates, sample_size)
            )

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


def get_digit_sum_statistics(past_draws: List[List[int]]) -> Dict:
    """
    Get comprehensive digit sum statistics.

    Returns a dictionary containing:
    - most_common: most common digit sums and their frequencies
    - min_sum: minimum digit sum
    - max_sum: maximum digit sum
    - mean_sum: average digit sum
    - sum_distribution: distribution of digit sums
    """
    if not past_draws:
        return {}

    all_sums = [digit_sum(num) for draw in past_draws for num in draw]
    if not all_sums:
        return {}

    # Calculate statistics
    stats = {
        'most_common': dict(analyze_digit_sums(past_draws).most_common(5)),
        'min_sum': min(all_sums),
        'max_sum': max(all_sums),
        'mean_sum': sum(all_sums) / len(all_sums),
        'sum_distribution': dict(Counter(all_sums))
    }

    return stats