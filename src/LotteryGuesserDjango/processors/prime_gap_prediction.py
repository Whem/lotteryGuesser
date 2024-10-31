# prime_gap_prediction.py

import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

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
    Generates lottery numbers based on prime gap prediction.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_needed=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_needed=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_needed: int,
    numbers_field: str
) -> List[int]:
    """Generate numbers based on prime gap prediction."""
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(numbers_field, flat=True)

    past_draws = list(past_draws_queryset)

    # Count frequency of prime gaps
    gap_counter = Counter()
    for draw in past_draws:
        if not isinstance(draw, list):
            continue
        sorted_draw = sorted(draw)
        gaps = [sorted_draw[i + 1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
        for gap in gaps:
            if is_prime(gap):
                gap_counter[gap] += 1

    # Find the most common prime gaps
    most_common_gaps = [gap for gap, count in gap_counter.most_common()]

    # Generate numbers based on most common prime gaps
    selected_numbers = []
    if most_common_gaps:
        # Start with a random starting number within valid range
        max_start = max_num - sum(most_common_gaps)
        if max_start < min_num:
            max_start = min_num
        start_number = random.randint(min_num, max_start)
        selected_numbers = [start_number]
        for gap in most_common_gaps:
            next_number = selected_numbers[-1] + gap
            if next_number <= max_num:
                selected_numbers.append(next_number)
            if len(selected_numbers) == total_needed:
                break
        # If not enough numbers, fill with random primes
        if len(selected_numbers) < total_needed:
            primes_in_range = [n for n in range(min_num, max_num + 1) if is_prime(n)]
            remaining_numbers = list(set(primes_in_range) - set(selected_numbers))
            random.shuffle(remaining_numbers)
            selected_numbers.extend(remaining_numbers[:total_needed - len(selected_numbers)])
    else:
        # If no prime gaps found, select random prime numbers
        primes_in_range = [n for n in range(min_num, max_num + 1) if is_prime(n)]
        if len(primes_in_range) >= total_needed:
            selected_numbers = random.sample(primes_in_range, total_needed)
        else:
            selected_numbers = random.sample(range(min_num, max_num + 1), total_needed)

    # Ensure unique numbers and correct count
    selected_numbers = list(set(selected_numbers))
    if len(selected_numbers) < total_needed:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_needed - len(selected_numbers)])
    elif len(selected_numbers) > total_needed:
        selected_numbers = selected_numbers[:total_needed]

    # Convert numbers to standard Python int
    selected_numbers = [int(num) for num in selected_numbers]

    # Sort and return the selected numbers
    return sorted(selected_numbers)
