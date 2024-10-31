# number_rarity_prediction.py
from collections import Counter
import random
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using rarity prediction.
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
    """Generate a set of numbers using rarity prediction."""
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

    number_stats = calculate_number_statistics(past_numbers, min_num, max_num)
    rarest_numbers = find_rarest_numbers(
        number_stats,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(rarest_numbers)


def calculate_number_statistics(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict[int, Dict[str, float]]:
    """
    Calculate comprehensive statistics about number appearances.
    Returns a dictionary with frequency and recency metrics for each number.
    """
    number_stats = {num: {
        'frequency': 0,
        'recent_frequency': 0,
        'draws_since_last': float('inf'),
        'average_gap': 0
    } for num in range(min_num, max_num + 1)}

    total_draws = len(past_draws)
    if total_draws == 0:
        return number_stats

    # Calculate frequency and recency statistics
    for draw_idx, draw in enumerate(past_draws):
        draw_weight = 1 + (draw_idx / total_draws)  # Recent draws weighted higher

        for num in draw:
            if min_num <= num <= max_num:
                number_stats[num]['frequency'] += 1

                # Recent frequency (last 20 draws)
                if draw_idx < 20:
                    number_stats[num]['recent_frequency'] += draw_weight

                # Update draws since last appearance
                if number_stats[num]['draws_since_last'] == float('inf'):
                    number_stats[num]['draws_since_last'] = draw_idx

    # Calculate normalized statistics and average gaps
    for num in range(min_num, max_num + 1):
        stats = number_stats[num]

        # Normalize frequencies
        stats['frequency'] /= total_draws
        stats['recent_frequency'] /= min(20, total_draws)

        # Calculate average gap between appearances
        appearances = []
        last_seen = None
        for draw_idx, draw in enumerate(past_draws):
            if num in draw:
                if last_seen is not None:
                    appearances.append(draw_idx - last_seen)
                last_seen = draw_idx

        if appearances:
            stats['average_gap'] = sum(appearances) / len(appearances)
        else:
            stats['average_gap'] = total_draws  # Never appeared

    return number_stats


def find_rarest_numbers(
        number_stats: Dict[int, Dict[str, float]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Find rare numbers using multiple metrics."""
    rarity_scores = {}

    # Calculate composite rarity score for each number
    for num in range(min_num, max_num + 1):
        stats = number_stats[num]

        # Combine multiple factors into rarity score
        # Lower score = rarer number
        rarity_score = (
                stats['frequency'] * 0.4 +
                stats['recent_frequency'] * 0.3 +
                (1 / (stats['draws_since_last'] + 1)) * 0.2 +
                (1 / (stats['average_gap'] + 1)) * 0.1
        )
        rarity_scores[num] = rarity_score

    # Select the rarest numbers
    sorted_numbers = sorted(rarity_scores.items(), key=lambda x: x[1])
    rarest_numbers = [number for number, _ in sorted_numbers[:required_numbers]]

    # If we don't have enough numbers, add random ones
    while len(rarest_numbers) < required_numbers:
        number = random.randint(min_num, max_num)
        if number not in rarest_numbers:
            rarest_numbers.append(number)

    return sorted(rarest_numbers)


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))