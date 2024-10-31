# number_position_frequency_prediction.py
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using position frequency prediction.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate a set of numbers using position frequency prediction."""
    # Get past draws
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:200])

    # Extract appropriate number sets from past draws
    if is_main:
        past_numbers = [draw.lottery_type_number for draw in past_draws
                        if isinstance(draw.lottery_type_number, list)]
    else:
        past_numbers = [draw.additional_numbers for draw in past_draws
                        if hasattr(draw, 'additional_numbers') and
                        isinstance(draw.additional_numbers, list)]

    if not past_numbers:
        return generate_random_numbers(min_num, max_num, required_numbers)

    position_frequencies = calculate_position_frequencies(past_numbers, required_numbers)
    predicted_numbers = generate_numbers_from_frequencies(
        position_frequencies,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def calculate_position_frequencies(
        past_draws: List[List[int]],
        required_numbers: int
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """
    Calculate comprehensive frequency statistics for each position.
    Returns a dictionary with recent and overall frequencies for each position.
    """
    frequencies = defaultdict(lambda: {
        'recent': defaultdict(float),  # Recent draws weighted higher
        'overall': defaultdict(float)  # All-time frequencies
    })

    if not past_draws:
        return frequencies

    # Calculate weights for recent draws
    total_draws = len(past_draws)
    for draw_idx, draw in enumerate(past_draws):
        sorted_draw = sorted(draw)
        recency_weight = 1 + (draw_idx / total_draws)  # More recent draws have higher weight

        for position, number in enumerate(sorted_draw):
            if position < required_numbers:  # Only consider needed positions
                # Update overall frequencies
                frequencies[position]['overall'][number] += 1

                # Update recent frequencies with weight
                if draw_idx < 20:  # Consider only last 20 draws for recent
                    frequencies[position]['recent'][number] += recency_weight

    # Normalize frequencies
    for position in frequencies:
        for timing in ['recent', 'overall']:
            total = sum(frequencies[position][timing].values())
            if total > 0:
                for number in frequencies[position][timing]:
                    frequencies[position][timing][number] /= total

    return frequencies


def generate_numbers_from_frequencies(
        frequencies: Dict[int, Dict[str, Dict[int, float]]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on position frequencies with improved selection."""
    predicted_numbers: Set[int] = set()

    if not frequencies:
        return generate_random_numbers(min_num, max_num, required_numbers)

    # First pass: Try to select numbers based on combined frequency scores
    for position in range(required_numbers):
        if position in frequencies:
            # Combine recent and overall frequencies with weights
            combined_scores = defaultdict(float)
            for num in range(min_num, max_num + 1):
                recent_score = frequencies[position]['recent'].get(num, 0) * 0.7
                overall_score = frequencies[position]['overall'].get(num, 0) * 0.3
                combined_scores[num] = recent_score + overall_score

            # Sort numbers by score for this position
            sorted_numbers = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Try to find the highest-scoring number that hasn't been used
            for number, _ in sorted_numbers:
                if number not in predicted_numbers:
                    predicted_numbers.add(number)
                    break

    # Second pass: Fill any remaining slots based on overall frequencies
    all_frequencies = defaultdict(float)
    for pos_freq in frequencies.values():
        for num, freq in pos_freq['overall'].items():
            all_frequencies[num] += freq

    while len(predicted_numbers) < required_numbers:
        available_numbers = set(range(min_num, max_num + 1)) - predicted_numbers
        if not available_numbers:
            break

        # Select highest frequency unused number
        next_number = max(
            available_numbers,
            key=lambda x: all_frequencies.get(x, 0)
        )
        predicted_numbers.add(next_number)

    # Final pass: Fill any remaining slots with random numbers
    while len(predicted_numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        predicted_numbers.add(new_number)

    return sorted(list(predicted_numbers))


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))