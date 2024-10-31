# pattern_matching_prediction.py
from collections import Counter, defaultdict
import random
from itertools import combinations
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using pattern matching prediction.
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
    """Generate a set of numbers using pattern matching prediction."""
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

    pattern_stats = analyze_patterns(past_numbers, min_num, max_num)
    selected_numbers = generate_numbers_from_patterns(
        pattern_stats,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(selected_numbers)


def analyze_patterns(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict[str, Dict]:
    """
    Analyze comprehensive patterns in past draws.
    Returns statistics about different types of patterns.
    """
    pattern_stats = {
        'differences': Counter(),  # Differences between consecutive numbers
        'spacings': Counter(),  # Regular spacing patterns
        'sequences': Counter(),  # Sequential patterns
        'clusters': defaultdict(Counter),  # Number clustering patterns
        'positional': defaultdict(Counter)  # Position-based patterns
    }

    if not past_draws:
        return pattern_stats

    # Analyze each draw
    for draw_idx, draw in enumerate(past_draws):
        sorted_draw = sorted(draw)

        # Calculate differences pattern
        differences = tuple(sorted_draw[i + 1] - sorted_draw[i]
                            for i in range(len(sorted_draw) - 1))
        pattern_stats['differences'][differences] += 1

        # Analyze spacing patterns
        spacings = set()
        for i, j in combinations(range(len(sorted_draw)), 2):
            spacing = sorted_draw[j] - sorted_draw[i]
            spacings.add(spacing)
        pattern_stats['spacings'].update(spacings)

        # Analyze sequential patterns
        for i in range(len(sorted_draw) - 2):
            seq = tuple(sorted_draw[j + 1] - sorted_draw[j]
                        for j in range(i, i + 2))
            pattern_stats['sequences'][seq] += 1

        # Analyze clustering
        for i in range(len(sorted_draw) - 1):
            cluster_size = 2
            while i + cluster_size <= len(sorted_draw):
                cluster = tuple(sorted_draw[i:i + cluster_size])
                pattern_stats['clusters'][cluster_size][cluster] += 1
                cluster_size += 1

        # Analyze positional patterns
        for pos, num in enumerate(sorted_draw):
            pattern_stats['positional'][pos][num] += 1

    return pattern_stats


def generate_numbers_from_patterns(
        pattern_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on analyzed patterns."""
    selected_numbers: Set[int] = set()

    # Try different pattern-based generation methods
    generation_methods = [
        lambda: generate_from_differences(pattern_stats, min_num, max_num, required_numbers),
        lambda: generate_from_spacings(pattern_stats, min_num, max_num, required_numbers),
        lambda: generate_from_sequences(pattern_stats, min_num, max_num, required_numbers),
        lambda: generate_from_clusters(pattern_stats, min_num, max_num, required_numbers),
        lambda: generate_from_positions(pattern_stats, min_num, max_num, required_numbers)
    ]

    # Try each method until we have enough numbers
    for method in generation_methods:
        if len(selected_numbers) >= required_numbers:
            break

        try:
            numbers = method()
            selected_numbers.update(numbers)
        except Exception:
            continue

    # Ensure we have exactly the required number of unique numbers
    selected_numbers = set(sorted(selected_numbers)[:required_numbers])

    # Fill any remaining slots with random numbers
    while len(selected_numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in selected_numbers:
            selected_numbers.add(num)

    return sorted(list(selected_numbers))


def generate_from_differences(
        pattern_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on difference patterns."""
    if not pattern_stats['differences']:
        return []

    most_common_pattern = pattern_stats['differences'].most_common(1)[0][0]

    # Try different starting numbers
    for attempt in range(10):
        max_start = max_num - sum(most_common_pattern)
        if max_start < min_num:
            max_start = min_num

        start_number = random.randint(min_num, max_start)
        numbers = [start_number]

        for diff in most_common_pattern:
            next_number = numbers[-1] + diff
            if min_num <= next_number <= max_num:
                numbers.append(next_number)
            else:
                break

        if len(numbers) == required_numbers:
            return numbers

    return []


def generate_from_spacings(
        pattern_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on spacing patterns."""
    if not pattern_stats['spacings']:
        return []

    common_spacings = [spacing for spacing, _ in
                       pattern_stats['spacings'].most_common(3)]

    for spacing in common_spacings:
        start_number = random.randint(min_num, max_num - spacing * (required_numbers - 1))
        numbers = [start_number + spacing * i for i in range(required_numbers)]

        if all(min_num <= num <= max_num for num in numbers):
            return numbers

    return []


def generate_from_sequences(
        pattern_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on sequential patterns."""
    if not pattern_stats['sequences']:
        return []

    numbers = set()
    common_sequences = pattern_stats['sequences'].most_common(5)

    for sequence, _ in common_sequences:
        start_number = random.randint(min_num, max_num - sum(sequence))
        current = start_number

        for diff in sequence:
            next_num = current + diff
            if min_num <= next_num <= max_num:
                numbers.add(next_num)
            current = next_num

    return list(numbers)


def generate_from_clusters(
        pattern_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on clustering patterns."""
    numbers = set()

    for cluster_size, clusters in pattern_stats['clusters'].items():
        if clusters:
            common_cluster = clusters.most_common(1)[0][0]
            numbers.update(common_cluster)

    return list(numbers)


def generate_from_positions(
        pattern_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on positional patterns."""
    numbers = set()

    for position, counts in pattern_stats['positional'].items():
        if counts:
            common_number = counts.most_common(1)[0][0]
            if min_num <= common_number <= max_num:
                numbers.add(common_number)

    return list(numbers)


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))