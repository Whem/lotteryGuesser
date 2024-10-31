# quantum_inspired_fractal_pattern_predictor.py

import random
import math
import cmath
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def quantum_inspired_random(min_val, max_val, n):
    """Generate quantum-inspired random numbers."""
    phase = [random.random() * 2 * math.pi for _ in range(n)]
    amplitude = [random.random() for _ in range(n)]
    quantum_state = [cmath.rect(amp, phi) for amp, phi in zip(amplitude, phase)]
    prob = [abs(state) ** 2 for state in quantum_state]
    total_prob = sum(prob)
    normalized_prob = [p / total_prob for p in prob]
    return [int(min_val + (max_val - min_val) * p) for p in normalized_prob]


def generate_fractal_sequence(seed, length, a=4, b=0.5, c=0.1):
    """Generate a fractal-like sequence using the logistic map with perturbation."""
    sequence = [seed]
    for _ in range(length - 1):
        next_val = a * sequence[-1] * (1 - sequence[-1]) + b * math.sin(2 * math.pi * sequence[-1]) + c * random.random()
        sequence.append(next_val % 1)  # Ensure the value stays within [0,1]
    return sequence


def fractal_dimension(sequence, eps=1e-10):
    """Calculate an approximation of the fractal dimension of the sequence."""
    n = len(sequence)
    diffs = [abs(sequence[i] - sequence[i-1]) for i in range(1, n)]
    log_diffs = [math.log(max(d, eps)) for d in diffs]
    return sum(log_diffs) / (n * math.log(1 / n))


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a quantum-inspired fractal pattern predictor.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        is_main=True
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count),
            is_main=False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        total_numbers: int,
        is_main: bool
) -> List[int]:
    """
    Generates a set of lottery numbers using quantum-inspired fractal pattern prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.
    - is_main: Boolean indicating whether this is for main numbers or additional numbers.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    try:
        # Retrieve past winning numbers
        if is_main:
            past_draws = list(
                lg_lottery_winner_number.objects.filter(
                    lottery_type=lottery_type_instance
                ).order_by('-id')[:100].values_list('lottery_type_number', flat=True)
            )
        else:
            past_draws = list(
                lg_lottery_winner_number.objects.filter(
                    lottery_type=lottery_type_instance
                ).order_by('-id')[:100].values_list('additional_numbers', flat=True)
            )

        if len(past_draws) < 10:
            # If not enough past draws, return quantum-inspired random numbers
            return sorted(quantum_inspired_random(min_num, max_num, total_numbers))

        # Flatten past draws and normalize
        flat_sequence = [num for draw in past_draws if isinstance(draw, list) for num in draw]
        normalized_sequence = [(num - min_num) / (max_num - min_num) for num in flat_sequence]

        # Calculate fractal dimension of past draws
        fd = fractal_dimension(normalized_sequence)

        # Generate fractal-like sequence
        seed = random.random()
        fractal_seq = generate_fractal_sequence(seed, total_numbers * 2, a=3.5 + fd)

        # Map fractal sequence to lottery number range
        mapped_numbers = [int(min_num + (max_num - min_num) * val) for val in fractal_seq]

        # Apply quantum-inspired perturbation
        quantum_numbers = quantum_inspired_random(min_num, max_num, total_numbers * 2)

        # Combine fractal and quantum numbers
        combined_numbers = [(f + q) // 2 for f, q in zip(mapped_numbers, quantum_numbers)]

        # Ensure uniqueness and correct range
        predicted_numbers = list(set(combined_numbers))
        predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

        # If not enough unique numbers, fill with quantum-inspired random selection
        if len(predicted_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            additional_numbers_needed = total_numbers - len(predicted_numbers)
            additional_numbers = quantum_inspired_random(min_num, max_num, additional_numbers_needed)
            # Ensure we only add numbers that are in the remaining set
            additional_numbers = [num for num in additional_numbers if num in remaining]
            predicted_numbers += additional_numbers

            # If still not enough, fill with random choices from remaining numbers
            if len(predicted_numbers) < total_numbers:
                remaining = remaining - set(predicted_numbers)
                predicted_numbers += random.sample(remaining, total_numbers - len(predicted_numbers))

        return sorted(predicted_numbers[:total_numbers])

    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error in quantum_inspired_fractal_pattern_predictor: {str(e)}")
        # Fall back to quantum-inspired random number generation
        return sorted(quantum_inspired_random(min_num, max_num, total_numbers))
