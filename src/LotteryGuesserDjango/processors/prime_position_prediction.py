# prime_position_prediction.py

import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from typing import List, Tuple

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on prime position prediction.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_numbers: int,
    numbers_field: str
) -> List[int]:
    """
    Generates lottery numbers based on prime position prediction.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(numbers_field, flat=True)
    past_draws = list(past_draws_queryset)

    # Initialize counters for each position
    position_prime_counts = [0] * total_numbers
    position_total_counts = [0] * total_numbers

    for draw in past_draws:
        if not isinstance(draw, list):
            continue
        if len(draw) != total_numbers:
            continue  # Skip draws that don't have the expected number of numbers
        for idx, number in enumerate(draw):
            try:
                num = int(number)
                if min_num <= num <= max_num:
                    position_total_counts[idx] += 1
                    if is_prime(num):
                        position_prime_counts[idx] += 1
            except (ValueError, TypeError):
                continue  # Skip invalid numbers

    # Determine for each position whether to place a prime or non-prime number
    position_is_prime = []
    for prime_count, total_count in zip(position_prime_counts, position_total_counts):
        if total_count == 0:
            position_is_prime.append(False)  # Default to non-prime if no data
        else:
            prime_ratio = prime_count / total_count
            # If primes appear more than 50% in this position, set as prime
            position_is_prime.append(prime_ratio > 0.5)

    # Generate numbers based on position prime status
    primes = [n for n in range(min_num, max_num + 1) if is_prime(n)]
    non_primes = [n for n in range(min_num, max_num + 1) if not is_prime(n)]

    selected_numbers = []
    for is_prime_position in position_is_prime:
        if is_prime_position and primes:
            number = random.choice(primes)
            primes.remove(number)
            selected_numbers.append(number)
        elif not is_prime_position and non_primes:
            number = random.choice(non_primes)
            non_primes.remove(number)
            selected_numbers.append(number)
        elif primes:
            # Fallback if no non-primes are left
            number = random.choice(primes)
            primes.remove(number)
            selected_numbers.append(number)
        elif non_primes:
            # Fallback if no primes are left
            number = random.choice(non_primes)
            non_primes.remove(number)
            selected_numbers.append(number)

    # Ensure unique numbers
    selected_numbers = list(set(selected_numbers))

    # If not enough numbers, fill with remaining numbers
    all_numbers = set(range(min_num, max_num + 1))
    remaining_numbers = list(all_numbers - set(selected_numbers))
    random.shuffle(remaining_numbers)
    while len(selected_numbers) < total_numbers and remaining_numbers:
        selected_numbers.append(remaining_numbers.pop())

    # Trim the list if we have too many numbers
    selected_numbers = selected_numbers[:total_numbers]

    # Convert numbers to standard Python int
    selected_numbers = [int(num) for num in selected_numbers]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
