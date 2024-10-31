# number_pair_affinity_prediction.py
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using pair affinity prediction.
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
    """Generate a set of numbers using pair affinity prediction."""
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

    pair_affinities = calculate_pair_affinities(past_numbers)
    predicted_numbers = generate_numbers_from_affinities(
        pair_affinities,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(set(predicted_numbers))[:required_numbers]


def calculate_pair_affinities(past_draws: List[List[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate affinity scores for number pairs based on their frequency
    and recency in past draws.
    """
    pair_counts = defaultdict(float)
    total_draws = len(past_draws)

    for draw_idx, draw in enumerate(past_draws):
        # Apply recency weight - more recent draws have higher weight
        recency_weight = 1 + (draw_idx / total_draws)

        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                pair = tuple(sorted([draw[i], draw[j]]))
                pair_counts[pair] += recency_weight

    return pair_counts


def generate_numbers_from_affinities(
        pair_affinities: Dict[Tuple[int, int], float],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on pair affinities with improved selection."""
    if not pair_affinities:
        return generate_random_numbers(min_num, max_num, required_numbers)

    predicted_numbers: Set[int] = set()
    # Sort pairs by affinity score
    sorted_pairs = sorted(pair_affinities.items(), key=lambda x: x[1], reverse=True)

    # First pass: Add numbers from high-affinity pairs
    for pair, _ in sorted_pairs:
        predicted_numbers.update(pair)
        if len(predicted_numbers) >= required_numbers:
            break

    # Second pass: If we need more numbers, add them based on individual number frequencies
    if len(predicted_numbers) < required_numbers:
        number_frequencies = defaultdict(float)
        for (num1, num2), affinity in pair_affinities.items():
            number_frequencies[num1] += affinity
            number_frequencies[num2] += affinity

        available_numbers = set(range(min_num, max_num + 1)) - predicted_numbers
        sorted_numbers = sorted(available_numbers,
                                key=lambda x: number_frequencies.get(x, 0),
                                reverse=True)

        for num in sorted_numbers:
            if len(predicted_numbers) >= required_numbers:
                break
            predicted_numbers.add(num)

    # Final pass: If we still need more numbers, add random ones
    while len(predicted_numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        predicted_numbers.add(new_number)

    return sorted(list(predicted_numbers))[:required_numbers]


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))