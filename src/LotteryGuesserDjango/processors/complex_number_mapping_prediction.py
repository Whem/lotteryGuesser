# complex_number_mapping_prediction.py
import cmath
from collections import Counter
from typing import List, Tuple
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Complex number mapping predictor for combined lottery types.
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
    """Generate a set of numbers using complex number mapping."""
    # Get past draws
    past_draws = get_historical_data(lottery_type_instance, is_main)

    # Create complex number mapping
    complex_map = create_complex_mapping(past_draws, required_numbers)

    if not complex_map:
        return generate_random_numbers(min_num, max_num, required_numbers)

    # Calculate centroid
    centroid = calculate_centroid(complex_map)

    # Generate predictions
    predicted_numbers = generate_predictions(
        complex_map,
        centroid,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


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


def create_complex_mapping(past_draws: List[List[int]], required_numbers: int) -> Counter:
    """Map numbers to complex plane points."""
    complex_map = Counter()

    for draw in past_draws:
        for i, number in enumerate(draw):
            # Map each number to a point on the complex plane
            angle = 2 * cmath.pi * i / required_numbers
            complex_number = cmath.rect(number, angle)
            complex_map[complex_number] += 1

    return complex_map


def calculate_centroid(complex_map: Counter) -> complex:
    """Calculate the centroid of all mapped points."""
    if not complex_map:
        return complex(0, 0)

    return sum(num * count for num, count in complex_map.items()) / sum(complex_map.values())


def generate_predictions(
        complex_map: Counter,
        centroid: complex,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions based on complex mapping."""
    predicted_numbers = []
    used_map = Counter(complex_map)

    while len(predicted_numbers) < required_numbers and used_map:
        # Find closest mapped number to centroid
        closest_complex = min(used_map, key=lambda x: abs(x - centroid))

        # Convert to valid integer
        predicted_number = max(min_num, min(max_num, int(abs(closest_complex))))

        if predicted_number not in predicted_numbers:
            predicted_numbers.append(predicted_number)

        # Remove used number
        del used_map[closest_complex]

    # Fill remaining with random numbers if needed
    while len(predicted_numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        if new_number not in predicted_numbers:
            predicted_numbers.append(new_number)

    return predicted_numbers


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers within range."""
    return random.sample(range(min_num, max_num + 1), required_numbers)