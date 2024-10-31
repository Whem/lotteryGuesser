# markov_chain_probability_prediction.py
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Generate lottery numbers using Markov Chain probability analysis."""
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

    return sorted(main_numbers), sorted(additional_numbers)


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using Markov Chain probability analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    # Build transition matrix
    transition_matrix = build_transition_matrix(
        past_draws,
        min_num,
        max_num
    )

    # Generate predictions
    predicted_numbers = generate_predictions(
        transition_matrix,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    field = 'lottery_type_number' if is_main else 'additional_numbers'
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(field, flat=True))

    return [draw for draw in past_draws if isinstance(draw, list)]


def build_transition_matrix(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        epsilon: float = 0.01
) -> Dict[int, Dict[int, float]]:
    """Build Markov chain transition matrix."""
    # Initialize matrix
    matrix = defaultdict(lambda: defaultdict(float))

    # Count transitions
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            matrix[sorted_draw[i]][sorted_draw[i + 1]] += 1

    # Normalize probabilities
    for num in matrix:
        total = sum(matrix[num].values())
        if total > 0:
            for next_num in matrix[num]:
                matrix[num][next_num] /= total

    # Add small probability for unseen transitions
    for i in range(min_num, max_num + 1):
        if i not in matrix:
            matrix[i] = defaultdict(lambda: epsilon)
        else:
            for j in range(min_num, max_num + 1):
                if j not in matrix[i]:
                    matrix[i][j] = epsilon

    return matrix


def generate_predictions(
        matrix: Dict[int, Dict[int, float]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions using Markov chain."""
    predicted_numbers = set()
    current_number = random.randint(min_num, max_num)

    while len(predicted_numbers) < required_numbers:
        # Add current number if unique
        if current_number not in predicted_numbers:
            predicted_numbers.add(current_number)

        # Choose next number
        current_number = choose_next_number(
            matrix,
            current_number,
            min_num,
            max_num
        )

    return list(predicted_numbers)[:required_numbers]


def choose_next_number(
        matrix: Dict[int, Dict[int, float]],
        current: int,
        min_num: int,
        max_num: int
) -> int:
    """Choose next number based on transition probabilities."""
    if current not in matrix:
        return random.randint(min_num, max_num)

    try:
        probabilities = list(matrix[current].items())
        numbers, probs = zip(*probabilities)
        return random.choices(numbers, weights=probs)[0]
    except Exception as e:
        print(f"Error choosing next number: {str(e)}")
        return random.randint(min_num, max_num)


def calculate_matrix_statistics(
        matrix: Dict[int, Dict[int, float]],
        min_num: int,
        max_num: int
) -> Dict[str, float]:
    """Calculate statistics about the transition matrix."""
    stats = {
        'avg_transition_prob': 0.0,
        'max_transition_prob': 0.0,
        'num_transitions': 0
    }

    try:
        probs = []
        for i in range(min_num, max_num + 1):
            if i in matrix:
                probs.extend(matrix[i].values())

        if probs:
            stats.update({
                'avg_transition_prob': sum(probs) / len(probs),
                'max_transition_prob': max(probs),
                'num_transitions': len(probs)
            })

    except Exception as e:
        print(f"Error calculating matrix statistics: {str(e)}")

    return stats