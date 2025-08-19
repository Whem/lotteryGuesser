# quantum_rng_simulation.py

from typing import List, Tuple
from algorithms.models import lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a quantum RNG simulation.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count
        )

    return main_numbers, additional_numbers


def generate_number_set(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Deterministically generates a set of lottery numbers using a low-discrepancy
    sequence (Van der Corput) to simulate uniform coverage without randomness.
    """
    span = max_num - min_num + 1
    if total_numbers <= 0 or span <= 0:
        return []
    picks = []
    used = set()
    i = 1  # 1-based index for Van der Corput
    while len(picks) < total_numbers and i < 100000:
        u = _van_der_corput(i, base=2)
        idx = int(u * span)
        if idx >= span:
            idx = span - 1
        candidate = min_num + idx
        if candidate not in used:
            picks.append(candidate)
            used.add(candidate)
        i += 1
    # Fill deterministically with smallest remaining if needed
    if len(picks) < total_numbers:
        for candidate in range(min_num, max_num + 1):
            if candidate not in used:
                picks.append(candidate)
                used.add(candidate)
                if len(picks) >= total_numbers:
                    break
    return sorted(picks)[:total_numbers]


def _van_der_corput(n: int, base: int = 2) -> float:
    """Van der Corput sequence for n (1-based) in the given base."""
    vdc = 0.0
    denom = 1.0
    while n > 0:
        n, remainder = divmod(n, base)
        denom *= base
        vdc += remainder / denom
    return vdc
