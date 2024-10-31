# number_grouping_prediction.py
from collections import defaultdict, Counter
import random
from typing import List, Dict, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using group prediction.
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
    """Generate a set of numbers using group prediction."""
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

    group_patterns = analyze_group_patterns(past_numbers, min_num, max_num)
    predicted_numbers = generate_numbers_from_patterns(
        group_patterns,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def analyze_group_patterns(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Counter:
    """Analyze patterns in number groups from historical draws."""
    group_size = (max_num - min_num + 1) // 3
    groups = {
        'low': range(min_num, min_num + group_size),
        'mid': range(min_num + group_size, min_num + 2 * group_size),
        'high': range(min_num + 2 * group_size, max_num + 1)
    }

    patterns = Counter()
    for draw in past_draws:
        pattern = ''.join(sorted([get_group(num, groups) for num in draw]))
        patterns[pattern] += 1

    return patterns


def get_group(number: int, groups: Dict[str, range]) -> str:
    """Determine which group a number belongs to."""
    for group, range_ in groups.items():
        if number in range_:
            return group[0]  # Return first letter of group name
    return 'x'  # Should never happen


def generate_numbers_from_patterns(
        patterns: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on analyzed patterns."""
    group_size = (max_num - min_num + 1) // 3
    groups = {
        'l': range(min_num, min_num + group_size),
        'm': range(min_num + group_size, min_num + 2 * group_size),
        'h': range(min_num + 2 * group_size, max_num + 1)
    }

    # Handle empty patterns case
    if not patterns:
        return generate_random_numbers(min_num, max_num, required_numbers)

    most_common_pattern = patterns.most_common(1)[0][0]
    predicted_numbers = []

    # Generate numbers based on pattern
    for group in most_common_pattern:
        if group in groups:
            predicted_numbers.append(random.choice(list(groups[group])))

    # Fill remaining slots if needed
    while len(predicted_numbers) < required_numbers:
        predicted_numbers.append(random.randint(min_num, max_num))

    # Ensure uniqueness and correct length
    unique_numbers = list(set(predicted_numbers))

    # If we lost too many numbers due to duplicates, add more
    while len(unique_numbers) < required_numbers:
        new_num = random.randint(min_num, max_num)
        if new_num not in unique_numbers:
            unique_numbers.append(new_num)

    return unique_numbers[:required_numbers]


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))