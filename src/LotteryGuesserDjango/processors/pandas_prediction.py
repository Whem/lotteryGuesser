# pandas_prediction.py
import random
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using pandas-based prediction.
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
    """Generate a set of numbers using pandas-based prediction."""
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

    # Convert to DataFrame and analyze patterns
    analysis_results = analyze_number_patterns(
        past_numbers,
        min_num,
        max_num,
        required_numbers
    )

    selected_numbers = generate_numbers_from_analysis(
        analysis_results,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(selected_numbers)


def analyze_number_patterns(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> Dict[str, pd.DataFrame]:
    """
    Analyze number patterns using pandas DataFrame operations.
    Returns various analytical results in DataFrames.
    """
    # Convert past draws to DataFrame
    df = pd.DataFrame(past_draws)

    analysis = {}

    # Transition frequency matrix
    transition_matrix = pd.DataFrame(0,
                                     index=range(min_num, max_num + 1),
                                     columns=range(min_num, max_num + 1))

    for i in range(len(df) - 1):
        current_row = df.iloc[i].dropna()
        next_row = df.iloc[i + 1].dropna()

        for current_num in current_row:
            for next_num in next_row:
                if min_num <= current_num <= max_num and min_num <= next_num <= max_num:
                    transition_matrix.loc[current_num, next_num] += 1

    analysis['transitions'] = transition_matrix

    # Position frequency analysis
    position_freq = pd.DataFrame(0,
                                 index=range(min_num, max_num + 1),
                                 columns=range(required_numbers))

    for pos in range(required_numbers):
        pos_series = df[pos].dropna()
        counts = pos_series.value_counts()
        for num, count in counts.items():
            if min_num <= num <= max_num:
                position_freq.loc[num, pos] = count

    analysis['positions'] = position_freq

    # Recent trends analysis
    recent_df = df.head(20)  # Last 20 draws
    recent_counts = pd.Series(0, index=range(min_num, max_num + 1))

    for _, row in recent_df.iterrows():
        numbers = row.dropna()
        for num in numbers:
            if min_num <= num <= max_num:
                recent_counts[num] += 1

    analysis['recent_trends'] = recent_counts.to_frame('count')

    # Moving averages and volatility
    rolling_counts = pd.DataFrame(index=range(min_num, max_num + 1))

    for window in [5, 10, 20]:
        counts = pd.Series(0, index=range(min_num, max_num + 1))
        for i in range(window):
            if i < len(df):
                numbers = df.iloc[i].dropna()
                for num in numbers:
                    if min_num <= num <= max_num:
                        counts[num] += 1
        rolling_counts[f'MA_{window}'] = counts / window

    analysis['moving_averages'] = rolling_counts

    return analysis


def generate_numbers_from_analysis(
        analysis_results: Dict[str, pd.DataFrame],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on pandas analysis results."""
    selected_numbers: Set[int] = set()

    # Calculate composite scores for each number
    scores = pd.Series(0.0, index=range(min_num, max_num + 1))

    # Weight from transition probabilities
    transitions = analysis_results['transitions']
    transition_scores = transitions.sum(axis=1) / transitions.sum().sum()
    scores += transition_scores * 0.3

    # Weight from positional frequencies
    positions = analysis_results['positions']
    position_scores = positions.sum(axis=1) / positions.sum().sum()
    scores += position_scores * 0.3

    # Weight from recent trends
    recent = analysis_results['recent_trends']
    recent_scores = recent['count'] / recent['count'].sum()
    scores += recent_scores * 0.2

    # Weight from moving averages
    ma_df = analysis_results['moving_averages']
    ma_scores = ma_df.mean(axis=1) / ma_df.mean().mean()
    scores += ma_scores * 0.2

    # First pass: Select best scoring numbers
    sorted_scores = scores.sort_values(ascending=False)
    top_numbers = sorted_scores.head(required_numbers // 2)
    selected_numbers.update(top_numbers.index)

    # Second pass: Fill remaining slots considering positions
    remaining_slots = required_numbers - len(selected_numbers)
    if remaining_slots > 0:
        position_weights = analysis_results['positions'].copy()
        # Zero out already selected numbers
        position_weights.loc[list(selected_numbers)] = 0

        for pos in range(remaining_slots):
            # Get best number for each remaining position
            pos_scores = position_weights[pos].copy()
            best_num = pos_scores.idxmax()
            if best_num not in selected_numbers:
                selected_numbers.add(best_num)

    # Final pass: Fill any remaining slots with random numbers
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