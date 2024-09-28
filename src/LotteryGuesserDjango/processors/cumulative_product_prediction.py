import random
from collections import Counter
from typing import List, Set
import numpy as np
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    cumulative_products = [np.prod(draw) for draw in past_draws]
    product_counter = Counter(cumulative_products)

    # Get the top 3 most common products
    common_products = product_counter.most_common(3)

    predicted_numbers = set()
    for target_product, _ in common_products:
        attempt = find_numbers_with_product(target_product, lottery_type_instance)
        if attempt:
            predicted_numbers.update(attempt)
            if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                break

    # If we don't have enough numbers, fill with random ones
    fill_remaining_numbers(predicted_numbers, lottery_type_instance)

    return sorted(list(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers])


def find_numbers_with_product(target_product: int, lottery_type_instance: lg_lottery_type, max_attempts: int = 1000) -> \
Set[int]:
    for _ in range(max_attempts):
        numbers = set()
        current_product = 1
        while current_product < target_product and len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
            new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
            if new_number not in numbers:
                numbers.add(new_number)
                current_product *= new_number

        if current_product == target_product:
            return numbers

    return set()  # If we couldn't find a matching set


def fill_remaining_numbers(numbers: Set[int], lottery_type_instance: lg_lottery_type):
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if new_number not in numbers:
            numbers.add(new_number)


def analyze_products(past_draws: List[List[int]], top_n: int = 5) -> None:
    products = [np.prod(draw) for draw in past_draws]
    product_counter = Counter(products)

    print(f"Top {top_n} most common products:")
    for product, count in product_counter.most_common(top_n):
        print(f"Product {product} occurred {count} times")

    print(f"\nProduct statistics:")
    print(f"Min product: {min(products)}")
    print(f"Max product: {max(products)}")
    print(f"Mean product: {np.mean(products):.2f}")
    print(f"Median product: {np.median(products):.2f}")

# Example usage of the analysis function:
# past_draws = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True))
# analyze_products(past_draws)