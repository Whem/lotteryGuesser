# ensemble_prediction.py
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

# Import get_numbers functions from existing predictor modules
from .fuzzy_logic_prediction import get_numbers as fuzzy_logic_get_numbers
from .cellular_automaton_prediction import get_numbers as cellular_automaton_get_numbers
from .topological_data_analysis_prediction import get_numbers as tda_get_numbers
from .symbolic_regression_prediction import get_numbers as symbolic_regression_get_numbers
from .graph_centrality_prediction import get_numbers as graph_centrality_get_numbers
from .chaotic_map_prediction import get_numbers as chaotic_map_get_numbers


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Ensemble predictor for combined lottery types.
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
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using ensemble prediction."""
    # Collect predictions from all models
    predictions = []

    # Gather predictions for the appropriate number set
    if is_main:
        predictions.extend([
            fuzzy_logic_get_numbers(lottery_type_instance)[0],
            cellular_automaton_get_numbers(lottery_type_instance)[0],
            tda_get_numbers(lottery_type_instance)[0],
            symbolic_regression_get_numbers(lottery_type_instance)[0],
            graph_centrality_get_numbers(lottery_type_instance)[0],
            chaotic_map_get_numbers(lottery_type_instance)[0]
        ])
    else:
        predictions.extend([
            fuzzy_logic_get_numbers(lottery_type_instance)[1],
            cellular_automaton_get_numbers(lottery_type_instance)[1],
            tda_get_numbers(lottery_type_instance)[1],
            symbolic_regression_get_numbers(lottery_type_instance)[1],
            graph_centrality_get_numbers(lottery_type_instance)[1],
            chaotic_map_get_numbers(lottery_type_instance)[1]
        ])

    # Count all predicted numbers
    all_predicted_numbers = [num for pred in predictions for num in pred]
    number_counts = Counter(all_predicted_numbers)

    # Select most common numbers within valid range
    predicted_numbers = []
    for num, _ in number_counts.most_common():
        if min_num <= num <= max_num:
            predicted_numbers.append(num)
            if len(predicted_numbers) >= required_numbers:
                break

    # Fill with historical numbers if needed
    if len(predicted_numbers) < required_numbers:
        historical_numbers = get_historical_numbers(
            lottery_type_instance,
            min_num,
            max_num,
            required_numbers,
            is_main,
            predicted_numbers
        )
        predicted_numbers.extend(historical_numbers)

    # Final sorting and trimming
    return sorted(predicted_numbers[:required_numbers])


def get_historical_numbers(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool,
        existing_numbers: List[int]
) -> List[int]:
    """Get additional numbers from historical data."""
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id')

    all_numbers = []
    for draw in past_draws:
        numbers = draw.lottery_type_number if is_main else draw.additional_numbers
        if isinstance(numbers, list):
            all_numbers.extend(numbers)

    # Count and filter numbers
    number_counts = Counter(all_numbers)
    additional_numbers = []

    for num, _ in number_counts.most_common():
        if (min_num <= num <= max_num and
                num not in existing_numbers and
                num not in additional_numbers):
            additional_numbers.append(num)
            if len(existing_numbers) + len(additional_numbers) >= required_numbers:
                break

    return additional_numbers