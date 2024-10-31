# odd_even_balance_prediction.py
from collections import Counter, defaultdict
import random
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using odd-even balance prediction.
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
    """Generate a set of numbers using odd-even balance prediction."""
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

    pattern_stats = analyze_odd_even_patterns(past_numbers, required_numbers)
    selected_numbers = generate_numbers_from_patterns(
        pattern_stats,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(selected_numbers)


def analyze_odd_even_patterns(
        past_draws: List[List[int]],
        required_numbers: int
) -> Dict[str, Dict]:
    """
    Analyze comprehensive odd-even patterns in past draws.
    Returns statistics about patterns and their effectiveness.
    """
    pattern_stats = {
        'distributions': Counter(),  # Counts of different odd-even distributions
        'positional': defaultdict(Counter),  # Odd-even preferences by position
        'sequences': Counter(),  # Consecutive odd-even patterns
        'effectiveness': defaultdict(list),  # How often each pattern appeared in winning numbers
        'trends': defaultdict(float)  # Recent trends in odd-even ratios
    }

    if not past_draws:
        return pattern_stats

    # Analyze each draw
    for draw_idx, draw in enumerate(past_draws):
        sorted_draw = sorted(draw)

        # Record distribution
        odd_count = sum(1 for num in draw if num % 2 != 0)
        even_count = len(draw) - odd_count
        pattern_stats['distributions'][(odd_count, even_count)] += 1

        # Analyze positional preferences
        for pos, num in enumerate(sorted_draw):
            if pos < required_numbers:
                pattern_stats['positional'][pos]['odd' if num % 2 != 0 else 'even'] += 1

        # Analyze sequences
        if len(sorted_draw) >= 2:
            sequence = ''.join('O' if num % 2 != 0 else 'E' for num in sorted_draw)
            pattern_stats['sequences'][sequence] += 1

        # Calculate recent trends
        if draw_idx < 20:  # Focus on recent draws
            ratio = odd_count / len(draw) if draw else 0
            pattern_stats['trends'][draw_idx] = ratio

    # Normalize and calculate effectiveness
    total_draws = len(past_draws)
    for pattern, count in pattern_stats['distributions'].items():
        pattern_stats['effectiveness'][pattern] = count / total_draws

    return pattern_stats


def generate_numbers_from_patterns(
        pattern_stats: Dict[str, Dict],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on analyzed odd-even patterns."""
    selected_numbers: Set[int] = set()

    # Get the most common and effective pattern
    if pattern_stats['distributions']:
        target_pattern = max(
            pattern_stats['distributions'].items(),
            key=lambda x: (x[1], pattern_stats['effectiveness'][x[0]])
        )[0]
    else:
        # Default to balanced distribution if no data
        target_pattern = (required_numbers // 2, required_numbers - required_numbers // 2)

    odd_count, even_count = target_pattern

    # Prepare number pools
    odd_numbers = [n for n in range(min_num, max_num + 1) if n % 2 != 0]
    even_numbers = [n for n in range(min_num, max_num + 1) if n % 2 == 0]

    # Score numbers based on positional preferences
    odd_scores = defaultdict(float)
    even_scores = defaultdict(float)

    for pos, counts in pattern_stats['positional'].items():
        total_pos = sum(counts.values())
        if total_pos > 0:
            odd_preference = counts['odd'] / total_pos
            even_preference = counts['even'] / total_pos

            # Apply positional preferences to scoring
            for num in odd_numbers:
                odd_scores[num] += odd_preference
            for num in even_numbers:
                even_scores[num] += even_preference

    # Select numbers based on scores and target pattern
    sorted_odds = sorted(odd_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_evens = sorted(even_scores.items(), key=lambda x: x[1], reverse=True)

    # Add highest scoring odd numbers
    for num, _ in sorted_odds[:odd_count]:
        selected_numbers.add(num)

    # Add highest scoring even numbers
    for num, _ in sorted_evens[:even_count]:
        selected_numbers.add(num)

    # Fill any remaining slots
    while len(selected_numbers) < required_numbers:
        remaining_odds = [n for n in odd_numbers if n not in selected_numbers]
        remaining_evens = [n for n in even_numbers if n not in selected_numbers]

        if random.random() < 0.5 and remaining_odds:
            selected_numbers.add(random.choice(remaining_odds))
        elif remaining_evens:
            selected_numbers.add(random.choice(remaining_evens))
        else:
            break

    # If still not enough numbers, add random ones
    while len(selected_numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        selected_numbers.add(num)

    return sorted(list(selected_numbers))


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))