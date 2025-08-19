# monte_carlo_simulation_prediction.py

import numpy as np
from typing import List, Tuple
from collections import defaultdict

from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using Monte Carlo simulation for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        num_picks=int(lottery_type_instance.pieces_of_draw_numbers),
        iterations=1000
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            num_picks=int(lottery_type_instance.additional_numbers_count),
            iterations=1000
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_numbers(min_num: int, max_num: int, num_picks: int, iterations: int = 1000) -> List[int]:
    """
    Generate numbers using Monte Carlo simulation.
    """
    frequency = monte_carlo_simulation(min_num, max_num, num_picks, iterations)

    most_common = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    predicted_numbers = set()
    for numbers, _ in most_common:
        predicted_numbers.update(numbers)
        if len(predicted_numbers) >= num_picks:
            break

    # Fill missing numbers deterministically with smallest remaining
    all_possible_numbers = set(range(min_num, max_num + 1))
    if len(predicted_numbers) < num_picks:
        for candidate in range(min_num, max_num + 1):
            if candidate not in predicted_numbers:
                predicted_numbers.add(candidate)
                if len(predicted_numbers) >= num_picks:
                    break

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    return sorted(predicted_numbers)[:num_picks]


def monte_carlo_simulation(min_num: int, max_num: int, num_picks: int, iterations: int) -> defaultdict:
    """
    Deterministic quasi–Monte Carlo using Halton sequence to generate combinations.
    """
    frequency = defaultdict(int)
    for i in range(iterations):
        numbers = tuple(sorted(_quasi_sample_combination(min_num, max_num, num_picks, i)))
        frequency[numbers] += 1
    return frequency


# --- Deterministic quasi–random utilities (Halton sequence) ---
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def _halton_value(index: int, base: int) -> float:
    """Return Halton sequence value for given 1-based index and base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _quasi_sample_combination(min_num: int, max_num: int, num_picks: int, seed_index: int) -> List[int]:
    """
    Deterministically sample a unique combination of num_picks numbers within range
    using Halton sequences across dimensions.
    """
    span = max_num - min_num + 1
    picks: List[int] = []
    used = set()
    for d in range(num_picks):
        base = _PRIMES[d % len(_PRIMES)]
        # 1-based index for Halton; vary by seed and dimension to avoid correlation
        u = _halton_value(seed_index + d + 1, base)
        idx = int(u * span)
        if idx >= span:
            idx = span - 1
        candidate = min_num + idx
        if candidate in used:
            # Find next available deterministically
            j = 1
            while True:
                next_candidate = min_num + ((idx + j) % span)
                if next_candidate not in used:
                    candidate = next_candidate
                    break
                j += 1
        picks.append(candidate)
        used.add(candidate)
    return picks
