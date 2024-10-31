# fibonacci_sequence_prediction.py
import random
from typing import List, Set, Tuple, Dict
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Fibonacci sequence predictor for combined lottery types.
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
    """Generate numbers using Fibonacci sequence analysis."""
    # Generate Fibonacci numbers
    fibonacci_numbers = generate_fibonacci_sequence(max_num)

    # Generate near-Fibonacci numbers
    near_fibonacci_numbers = generate_near_fibonacci_numbers(
        fibonacci_numbers,
        min_num,
        max_num
    )

    # Get historical data and analyze frequencies
    past_draws = get_historical_data(lottery_type_instance, is_main)
    frequency = analyze_frequencies(past_draws, near_fibonacci_numbers)

    # Generate predictions
    predicted_numbers = generate_predictions(
        near_fibonacci_numbers,
        frequency,
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


def generate_fibonacci_sequence(max_number: int) -> List[int]:
    """Generate Fibonacci sequence up to max_number."""
    sequence = [0, 1]
    while sequence[-1] < max_number:
        next_number = sequence[-1] + sequence[-2]
        if next_number > max_number:
            break
        sequence.append(next_number)
    return sequence[2:]  # Exclude 0 and 1


def generate_near_fibonacci_numbers(
        fibonacci_numbers: List[int],
        min_num: int,
        max_num: int,
        offset: int = 2
) -> Set[int]:
    """Generate numbers near Fibonacci numbers within range."""
    near_fibonacci = set()

    for fib in fibonacci_numbers:
        for i in range(-offset, offset + 1):
            number = fib + i
            if min_num <= number <= max_num:
                near_fibonacci.add(number)

    return near_fibonacci


def analyze_frequencies(
        past_draws: List[List[int]],
        near_fibonacci_numbers: Set[int]
) -> Dict[int, int]:
    """Analyze frequency of near-Fibonacci numbers in past draws."""
    frequency = {num: 0 for num in near_fibonacci_numbers}

    for draw in past_draws:
        for number in draw:
            if number in frequency:
                frequency[number] += 1

    return frequency


def generate_predictions(
        near_fibonacci_numbers: Set[int],
        frequency: Dict[int, int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions based on Fibonacci analysis."""
    predicted_numbers = set()

    # Sort numbers by frequency
    sorted_numbers = sorted(
        near_fibonacci_numbers,
        key=lambda x: frequency[x],
        reverse=True
    )

    # Add most frequent numbers
    predicted_numbers.update(sorted_numbers[:required_numbers])

    # Fill remaining numbers if needed
    while len(predicted_numbers) < required_numbers:
        # Try random near-Fibonacci number first
        if sorted_numbers:
            number = random.choice(sorted_numbers)
            if number not in predicted_numbers:
                predicted_numbers.add(number)
                continue

        # If no suitable near-Fibonacci numbers, use random number
        number = random.randint(min_num, max_num)
        if number not in predicted_numbers:
            predicted_numbers.add(number)

    return list(predicted_numbers)


def get_fibonacci_statistics(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict:
    """
    Get comprehensive Fibonacci statistics.

    Returns:
    - fibonacci_numbers: list of Fibonacci numbers in range
    - near_fibonacci_count: count of numbers near Fibonacci
    - percentage_near_fibonacci: percentage of numbers near Fibonacci
    - frequency_distribution: frequency of near-Fibonacci numbers
    """
    if not past_draws:
        return {}

    fibonacci_numbers = generate_fibonacci_sequence(max_num)
    near_fibonacci = generate_near_fibonacci_numbers(
        fibonacci_numbers,
        min_num,
        max_num
    )

    total_numbers = sum(len(draw) for draw in past_draws)
    near_count = sum(
        1 for draw in past_draws
        for num in draw
        if num in near_fibonacci
    )

    stats = {
        'fibonacci_numbers': fibonacci_numbers,
        'near_fibonacci_count': near_count,
        'total_numbers': total_numbers,
        'percentage_near_fibonacci': (near_count / total_numbers * 100) if total_numbers > 0 else 0,
        'frequency_distribution': analyze_frequencies(past_draws, near_fibonacci)
    }

    return stats