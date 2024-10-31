# harmonic_mean_prediction.py
import random
from typing import List, Tuple
from statistics import harmonic_mean
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Harmonic mean predictor for combined lottery types."""
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
    """Generate numbers using harmonic mean analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)
    all_numbers = [num for draw in past_draws for num in draw if num != 0]

    if not all_numbers:
        return random_selection(min_num, max_num, required_numbers)

    hm = harmonic_mean(all_numbers)
    std_dev = calculate_standard_deviation(all_numbers)

    predicted_numbers = generate_predictions(
        hm, std_dev, min_num, max_num, required_numbers
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


def generate_predictions(
        hm: float,
        std_dev: float,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions using statistical parameters."""
    predicted_numbers = set()
    attempts = 0
    max_attempts = required_numbers * 10

    while len(predicted_numbers) < required_numbers and attempts < max_attempts:
        number = generate_number(hm, std_dev, min_num, max_num)
        if number:
            predicted_numbers.add(number)
        attempts += 1

    if len(predicted_numbers) < required_numbers:
        remaining = required_numbers - len(predicted_numbers)
        predicted_numbers.update(
            random_selection(min_num, max_num, remaining)
        )

    return sorted(predicted_numbers)


def generate_number(hm: float, std_dev: float, min_num: int, max_num: int) -> int:
    """Generate single number using statistical distribution."""
    number = int(hm + random.gauss(0, std_dev))
    if min_num <= number <= max_num:
        return number
    return 0


def calculate_standard_deviation(numbers: List[int]) -> float:
    """Calculate standard deviation of numbers."""
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return variance ** 0.5


def random_selection(min_num: int, max_num: int, count: int) -> List[int]:
    """Generate random number selection."""
    return random.sample(range(min_num, max_num + 1), count)