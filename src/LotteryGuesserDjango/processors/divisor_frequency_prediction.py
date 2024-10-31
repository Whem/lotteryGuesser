# divisor_frequency_prediction.py
from collections import Counter
from typing import List, Dict, Tuple, Set
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Divisor frequency predictor for combined lottery types.
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
    """Generate a set of numbers using divisor frequency analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    # Create divisor map
    divisor_map = create_divisor_map(min_num, max_num)

    # Analyze divisor frequencies
    divisor_counter = analyze_divisor_frequencies(past_draws, divisor_map)

    # Generate predictions
    predicted_numbers = generate_predictions(
        divisor_counter,
        divisor_map,
        min_num,
        max_num,
        required_numbers
    )

    # Fill remaining if needed
    fill_remaining_numbers(predicted_numbers, min_num, max_num, required_numbers)

    return sorted(list(predicted_numbers))


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


def find_divisors(num: int) -> List[int]:
    """Find all divisors of a number."""
    divisors = []
    for i in range(1, int(num ** 0.5) + 1):
        if num % i == 0:
            divisors.append(i)
            if i != num // i:
                divisors.append(num // i)
    return sorted(divisors)


def create_divisor_map(min_num: int, max_num: int) -> Dict[int, List[int]]:
    """Create a map of numbers to their divisors."""
    return {
        num: find_divisors(num)
        for num in range(min_num, max_num + 1)
    }


def analyze_divisor_frequencies(
        past_draws: List[List[int]],
        divisor_map: Dict[int, List[int]]
) -> Counter:
    """Analyze divisor frequencies in past draws."""
    divisor_counter = Counter()
    for draw in past_draws:
        for number in draw:
            if number in divisor_map:
                divisor_counter.update(divisor_map[number])
    return divisor_counter


def generate_predictions(
        divisor_counter: Counter,
        divisor_map: Dict[int, List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> Set[int]:
    """Generate predictions based on common divisors."""
    predicted_numbers = set()

    # Get top N most common divisors
    N = min(10, required_numbers * 2)
    common_divisors = [div for div, _ in divisor_counter.most_common(N)]

    for div in common_divisors:
        candidates = [
            num for num in range(min_num, max_num + 1)
            if num in divisor_map and div in divisor_map[num]
        ]
        if candidates:
            predicted_numbers.add(random.choice(candidates))

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
        if new_number not in numbers:
            numbers.add(new_number)


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))


def get_divisor_statistics(past_draws: List[List[int]]) -> Dict:
    """
    Get comprehensive divisor statistics.

    Returns a dictionary containing:
    - most_common: most common divisors and their frequencies
    - min_divisor: minimum divisor
    - max_divisor: maximum divisor
    - mean_divisor: average divisor
    - divisor_distribution: distribution of divisors
    """
    if not past_draws:
        return {}

    divisor_counter = Counter()
    all_divisors = []

    for draw in past_draws:
        for number in draw:
            divisors = find_divisors(number)
            divisor_counter.update(divisors)
            all_divisors.extend(divisors)

    if not all_divisors:
        return {}

    stats = {
        'most_common': dict(divisor_counter.most_common(5)),
        'min_divisor': min(all_divisors),
        'max_divisor': max(all_divisors),
        'mean_divisor': sum(all_divisors) / len(all_divisors),
        'divisor_distribution': dict(Counter(all_divisors))
    }

    return stats