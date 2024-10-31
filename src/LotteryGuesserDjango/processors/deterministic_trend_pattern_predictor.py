# deterministic_trend_pattern_predictor.py
from collections import defaultdict, Counter
from django.apps import apps
import numpy as np
from typing import List, Tuple, Dict


def get_numbers(lottery_type_instance) -> Tuple[List[int], List[int]]:
    """
    Deterministic trend pattern predictor for combined lottery types.
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
        lottery_type_instance,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate a set of numbers using deterministic trend analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 10:
        return get_most_common_numbers(past_draws, min_num, max_num, required_numbers)

    # Analyze different patterns
    frequency = analyze_frequency(past_draws, min_num, max_num)
    recency = analyze_recency(past_draws, min_num, max_num)
    patterns = analyze_patterns(past_draws)
    trends = analyze_trends(past_draws, min_num, max_num)

    # Combine scores
    final_scores = combine_scores(
        frequency,
        recency,
        patterns,
        trends,
        min_num,
        max_num
    )

    # Select top scoring numbers
    return select_top_numbers(final_scores, min_num, required_numbers)


def get_historical_data(lottery_type_instance, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def analyze_frequency(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze number frequency in past draws."""
    frequency = Counter(num for draw in past_draws for num in draw)
    return [frequency.get(num, 0) for num in range(min_num, max_num + 1)]


def analyze_recency(past_draws: List[List[int]], min_num: int, max_num: int) -> List[float]:
    """Analyze how recently numbers appeared."""
    last_seen = defaultdict(lambda: float('inf'))
    for i, draw in enumerate(reversed(past_draws)):
        for num in draw:
            if last_seen[num] == float('inf'):
                last_seen[num] = i
    return [last_seen.get(num, float('inf')) for num in range(min_num, max_num + 1)]


def analyze_patterns(past_draws: List[List[int]]) -> Dict[int, int]:
    """Analyze sequential patterns in past draws."""
    patterns = defaultdict(int)
    for i in range(len(past_draws) - 1):
        for num in past_draws[i]:
            if num in past_draws[i + 1]:
                patterns[num] += 1
    return patterns


def analyze_trends(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze recent trends for each number."""
    trends = [[] for _ in range(min_num, max_num + 1)]
    for draw in past_draws:
        for num in range(min_num, max_num + 1):
            trends[num - min_num].append(1 if num in draw else 0)
    return [sum(trend[-5:]) for trend in trends]


def combine_scores(
        frequency: List[int],
        recency: List[float],
        patterns: Dict[int, int],
        trends: List[int],
        min_num: int,
        max_num: int
) -> np.ndarray:
    """Combine different scores into final scores."""
    # Normalize scores
    freq_score = np.array(frequency) / max(frequency) if max(frequency) > 0 else np.zeros_like(frequency)
    rec_score = 1 - (np.array(recency) / max(recency)) if max(recency) > 0 else np.zeros_like(recency)
    pat_score = np.array([patterns.get(num, 0) for num in range(min_num, max_num + 1)])
    pat_score = pat_score / max(pat_score) if max(pat_score) > 0 else np.zeros_like(pat_score)
    trend_score = np.array(trends) / 5

    # Weighted combination
    return (0.3 * freq_score +
            0.2 * rec_score +
            0.2 * pat_score +
            0.3 * trend_score)


def select_top_numbers(scores: np.ndarray, min_num: int, required_numbers: int) -> List[int]:
    """Select top scoring numbers."""
    top_indices = sorted(range(len(scores)),
                         key=lambda i: scores[i],
                         reverse=True)[:required_numbers]
    return sorted(num + min_num for num in top_indices)


def get_most_common_numbers(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Get most common numbers when insufficient data for analysis."""
    if not past_draws:
        return list(range(min_num, min_num + required_numbers))

    all_numbers = [num for draw in past_draws for num in draw]
    common_numbers = Counter(all_numbers).most_common(required_numbers)

    if len(common_numbers) < required_numbers:
        available = set(range(min_num, max_num + 1)) - {num for num, _ in common_numbers}
        additional = sorted(list(available))[:required_numbers - len(common_numbers)]
        common_numbers.extend((num, 0) for num in additional)

    return sorted(num for num, _ in common_numbers[:required_numbers])


def get_trend_statistics(past_draws: List[List[int]], min_num: int, max_num: int) -> Dict:
    """
    Get comprehensive trend statistics.

    Returns a dictionary containing:
    - frequency_stats
    - recency_stats
    - pattern_stats
    - trend_stats
    """
    stats = {
        'frequency': analyze_frequency(past_draws, min_num, max_num),
        'recency': analyze_recency(past_draws, min_num, max_num),
        'patterns': analyze_patterns(past_draws),
        'trends': analyze_trends(past_draws, min_num, max_num)
    }

    return stats