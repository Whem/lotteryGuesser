# cumulative_product_prediction.py
import random
from collections import Counter
from typing import List, Set, Tuple, Dict
import numpy as np
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Cumulative product predictor for combined lottery types.
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
    """Generate a set of numbers using cumulative product analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    # Calculate cumulative products
    products = analyze_cumulative_products(past_draws)

    # Generate predictions
    predicted_numbers = generate_predictions(
        products,
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


def analyze_cumulative_products(past_draws: List[List[int]]) -> Counter:
    """Analyze cumulative products from past draws."""
    products = [np.prod(draw) for draw in past_draws if draw]
    return Counter(products)


def generate_predictions(
        product_counter: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int,
        top_products: int = 3
) -> Set[int]:
    """Generate predictions based on common products."""
    predicted_numbers = set()

    # Get most common products
    common_products = product_counter.most_common(top_products)

    for target_product, _ in common_products:
        numbers = find_numbers_with_product(
            target_product,
            min_num,
            max_num,
            required_numbers
        )
        if numbers:
            predicted_numbers.update(numbers)
            if len(predicted_numbers) >= required_numbers:
                break

    return predicted_numbers


def find_numbers_with_product(
        target_product: int,
        min_num: int,
        max_num: int,
        required_numbers: int,
        max_attempts: int = 1000
) -> Set[int]:
    """Find a set of numbers with a target product."""
    for _ in range(max_attempts):
        numbers = set()
        current_product = 1

        while current_product < target_product and len(numbers) < required_numbers:
            new_number = random.randint(min_num, max_num)
            if new_number not in numbers:
                numbers.add(new_number)
                current_product *= new_number

        if current_product == target_product:
            return numbers

    return set()


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


def analyze_product_statistics(
        past_draws: List[List[int]]
) -> Dict[str, float]:
    """
    Analyze cumulative product statistics.

    Returns a dictionary containing:
    - min_product
    - max_product
    - mean_product
    - median_product
    - top_products (most common)
    """
    if not past_draws:
        return {}

    products = [np.prod(draw) for draw in past_draws if draw]
    if not products:
        return {}

    stats = {
        'min_product': float(min(products)),
        'max_product': float(max(products)),
        'mean_product': float(np.mean(products)),
        'median_product': float(np.median(products))
    }

    # Add top products
    product_counter = Counter(products)
    stats['top_products'] = dict(product_counter.most_common(5))

    return stats