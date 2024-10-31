# pair_frequency_prediction.py
from collections import Counter, defaultdict
import random
from itertools import combinations
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using pair frequency prediction.
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
    """Generate a set of numbers using pair frequency prediction."""
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

    pair_stats = analyze_pair_patterns(past_numbers, min_num, max_num)
    selected_numbers = generate_numbers_from_pairs(
        pair_stats,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(selected_numbers)


def analyze_pair_patterns(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict[str, Dict]:
    """
    Analyze comprehensive pair patterns in past draws.
    Returns statistics about pair frequencies and relationships.
    """
    pair_stats = {
        'frequency': Counter(),  # Raw pair frequencies
        'recent_frequency': Counter(),  # Recent pair frequencies
        'position_pairs': defaultdict(Counter),  # Position-based pair frequencies
        'number_connections': defaultdict(set),  # Numbers commonly drawn together
        'compatibility_score': defaultdict(float)  # How well numbers work together
    }

    if not past_draws:
        return pair_stats

    total_draws = len(past_draws)

    # Analyze each draw
    for draw_idx, draw in enumerate(past_draws):
        sorted_draw = sorted(draw)

        # Generate and count all pairs
        draw_pairs = list(combinations(sorted_draw, 2))

        # Update raw frequencies
        pair_stats['frequency'].update(draw_pairs)

        # Update recent frequencies with higher weight for recent draws
        if draw_idx < 20:  # Focus on recent draws
            weight = 1 + ((20 - draw_idx) / 20)  # Weight decreases with age
            for pair in draw_pairs:
                pair_stats['recent_frequency'][pair] += weight

        # Update position-based pairs
        for i, num1 in enumerate(sorted_draw[:-1]):
            for j, num2 in enumerate(sorted_draw[i + 1:], i + 1):
                pair_stats['position_pairs'][(i, j)][(num1, num2)] += 1

        # Update number connections
        for num1 in draw:
            for num2 in draw:
                if num1 != num2:
                    pair_stats['number_connections'][num1].add(num2)

    # Calculate compatibility scores
    all_numbers = set(range(min_num, max_num + 1))
    for num1 in all_numbers:
        for num2 in all_numbers:
            if num1 != num2:
                # Calculate score based on multiple factors
                raw_freq = pair_stats['frequency'].get((min(num1, num2), max(num1, num2)), 0)
                recent_freq = pair_stats['recent_frequency'].get((min(num1, num2), max(num1, num2)), 0)
                connection_strength = 1 if num2 in pair_stats['number_connections'][num1] else 0

                # Combine factors into compatibility score
                score = (
                        (raw_freq / total_draws) * 0.4 +
                        (recent_freq / 20) * 0.4 +
                        connection_strength * 0.2
                )
                pair_stats['compatibility_score'][(num1, num2)] = score

    return pair_stats


def generate_numbers_from_pairs(
        pair_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on analyzed pair patterns."""
    selected_numbers: Set[int] = set()
    number_scores = defaultdict(float)

    # Calculate initial scores for all numbers based on their pair relationships
    for (num1, num2), score in pair_stats['compatibility_score'].items():
        number_scores[num1] += score
        number_scores[num2] += score

    # First pass: Select numbers with highest pair compatibility
    sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
    initial_selections = {num for num, _ in sorted_numbers[:required_numbers // 2]}
    selected_numbers.update(initial_selections)

    # Second pass: Add numbers that work well with the initial selections
    remaining_slots = required_numbers - len(selected_numbers)
    if remaining_slots > 0:
        compatibility_with_selected = defaultdict(float)

        for num in range(min_num, max_num + 1):
            if num not in selected_numbers:
                score = sum(
                    pair_stats['compatibility_score'].get((min(num, sel), max(num, sel)), 0)
                    for sel in selected_numbers
                )
                compatibility_with_selected[num] = score

        # Add most compatible remaining numbers
        best_remaining = sorted(compatibility_with_selected.items(), key=lambda x: x[1], reverse=True)
        for num, _ in best_remaining:
            if len(selected_numbers) >= required_numbers:
                break
            selected_numbers.add(num)

    # Final pass: Fill any remaining slots with random numbers
    while len(selected_numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in selected_numbers:
            selected_numbers.add(num)

    return sorted(list(selected_numbers))


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))