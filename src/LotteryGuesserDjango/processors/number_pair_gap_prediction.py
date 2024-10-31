# number_pair_gap_prediction.py
from collections import defaultdict
import random
import statistics
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using pair gap prediction.
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
    """Generate a set of numbers using pair gap prediction."""
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

    pair_gaps = calculate_pair_gaps(past_numbers)
    predicted_numbers = generate_numbers_from_gaps(
        pair_gaps,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(set(predicted_numbers))[:required_numbers]


def calculate_pair_gaps(past_draws: List[List[int]]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Calculate gap statistics for number pairs including mean, median, and trend.
    Returns a dictionary with gap statistics for each pair.
    """
    pair_gaps = defaultdict(list)
    pair_last_seen = {}

    # Calculate basic gaps
    for draw_index, draw in enumerate(past_draws):
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                pair = tuple(sorted([draw[i], draw[j]]))
                if pair in pair_last_seen:
                    gap = draw_index - pair_last_seen[pair]
                    pair_gaps[pair].append(gap)
                pair_last_seen[pair] = draw_index

    # Calculate statistics for each pair
    pair_stats = {}
    for pair, gaps in pair_gaps.items():
        if gaps:  # Only process pairs that have appeared multiple times
            stats = {
                'mean': statistics.mean(gaps),
                'median': statistics.median(gaps),
                'last_gap': gaps[-1] if gaps else float('inf'),
                'trend': calculate_trend(gaps),
                'frequency': len(gaps)
            }
            pair_stats[pair] = stats

    return pair_stats


def calculate_trend(gaps: List[int]) -> float:
    """Calculate the trend in gap sizes (positive means increasing gaps)."""
    if len(gaps) < 2:
        return 0

    differences = [gaps[i + 1] - gaps[i] for i in range(len(gaps) - 1)]
    return sum(differences) / len(differences)


def generate_numbers_from_gaps(
        pair_stats: Dict[Tuple[int, int], Dict[str, float]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on pair gap statistics with improved selection."""
    if not pair_stats:
        return generate_random_numbers(min_num, max_num, required_numbers)

    predicted_numbers: Set[int] = set()

    # Score pairs based on multiple factors
    pair_scores = {}
    for pair, stats in pair_stats.items():
        # Lower score is better
        score = (
                stats['mean'] * 0.3 +  # Lower average gaps preferred
                stats['last_gap'] * 0.3 +  # Lower current gap preferred
                abs(stats['trend']) * 0.2 +  # Stable patterns preferred
                (1 / stats['frequency']) * 0.2  # Higher frequency preferred
        )
        pair_scores[pair] = score

    # First pass: Add numbers from pairs with best scores
    sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1])
    for pair, _ in sorted_pairs:
        predicted_numbers.update(pair)
        if len(predicted_numbers) >= required_numbers:
            break

    # Second pass: Add numbers based on individual frequency in pairs
    if len(predicted_numbers) < required_numbers:
        number_scores = defaultdict(float)
        for (num1, num2), score in pair_scores.items():
            number_scores[num1] += 1 / score
            number_scores[num2] += 1 / score

        available_numbers = set(range(min_num, max_num + 1)) - predicted_numbers
        sorted_numbers = sorted(available_numbers,
                                key=lambda x: number_scores.get(x, 0),
                                reverse=True)

        for num in sorted_numbers:
            if len(predicted_numbers) >= required_numbers:
                break
            predicted_numbers.add(num)

    # Final pass: Fill remaining slots with random numbers
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