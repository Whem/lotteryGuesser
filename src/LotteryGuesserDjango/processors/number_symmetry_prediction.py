# number_symmetry_prediction.py
from collections import Counter
import random
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using symmetry prediction.
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
    """Generate a set of numbers using symmetry prediction."""
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

    symmetry_stats = analyze_symmetry_patterns(past_numbers, min_num, max_num)
    selected_numbers = select_numbers_by_symmetry(
        symmetry_stats,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(selected_numbers)


def is_symmetric(number: int) -> bool:
    """Check if a number is symmetric (palindrome)."""
    number_str = str(number)
    return number_str == number_str[::-1]


def has_partial_symmetry(number: int) -> bool:
    """Check if a number has partial symmetry patterns."""
    number_str = str(number)

    # Check for repeating digits
    if len(set(number_str)) == 1:
        return True

    # Check for ascending/descending patterns
    digits = [int(d) for d in number_str]
    if all(digits[i] <= digits[i + 1] for i in range(len(digits) - 1)) or \
            all(digits[i] >= digits[i + 1] for i in range(len(digits) - 1)):
        return True

    # Check for alternating patterns
    if len(number_str) >= 2:
        evens = set(number_str[::2])
        odds = set(number_str[1::2])
        if len(evens) == 1 or len(odds) == 1:
            return True

    return False


def analyze_symmetry_patterns(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict[int, Dict[str, float]]:
    """
    Analyze symmetry patterns in past draws.
    Returns statistics about frequency and patterns for each number.
    """
    symmetry_stats = {num: {
        'frequency': 0,
        'recent_frequency': 0,
        'symmetry_score': 0,
        'last_appearance': float('inf')
    } for num in range(min_num, max_num + 1)}

    total_draws = len(past_draws)
    if total_draws == 0:
        return symmetry_stats

    # Analyze frequency and patterns
    for draw_idx, draw in enumerate(past_draws):
        draw_weight = 1 + (draw_idx / total_draws)  # Recent draws weighted higher

        for num in draw:
            if min_num <= num <= max_num:
                stats = symmetry_stats[num]

                # Update frequencies
                stats['frequency'] += 1
                if draw_idx < 20:  # Recent frequency
                    stats['recent_frequency'] += draw_weight

                # Update last appearance
                stats['last_appearance'] = min(stats['last_appearance'], draw_idx)

    # Calculate symmetry scores
    for num in range(min_num, max_num + 1):
        symmetry_score = 0
        if is_symmetric(num):
            symmetry_score = 1.0
        elif has_partial_symmetry(num):
            symmetry_score = 0.5

        symmetry_stats[num]['symmetry_score'] = symmetry_score

    # Normalize statistics
    for stats in symmetry_stats.values():
        stats['frequency'] = stats['frequency'] / total_draws
        stats['recent_frequency'] = stats['recent_frequency'] / min(20, total_draws)

    return symmetry_stats


def select_numbers_by_symmetry(
        symmetry_stats: Dict[int, Dict[str, float]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Select numbers based on symmetry patterns and historical performance."""
    selection_scores = {}

    # Calculate selection scores
    for num in range(min_num, max_num + 1):
        stats = symmetry_stats[num]

        # Combine factors into final score
        score = (
                stats['symmetry_score'] * 0.4 +
                stats['recent_frequency'] * 0.3 +
                stats['frequency'] * 0.2 +
                (1 / (stats['last_appearance'] + 1)) * 0.1
        )
        selection_scores[num] = score

    # Select numbers with best scores
    sorted_numbers = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)
    selected_numbers = [num for num, _ in sorted_numbers[:required_numbers]]

    # If we need more numbers, add random symmetric numbers
    if len(selected_numbers) < required_numbers:
        symmetric_numbers = [
            num for num in range(min_num, max_num + 1)
            if is_symmetric(num) and num not in selected_numbers
        ]
        random.shuffle(symmetric_numbers)
        selected_numbers.extend(symmetric_numbers[:required_numbers - len(selected_numbers)])

    # If still not enough, add random numbers
    while len(selected_numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in selected_numbers:
            selected_numbers.append(num)

    return selected_numbers


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))