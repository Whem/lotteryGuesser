# cellular_automaton_prediction.py
import numpy as np
from collections import Counter
from typing import List, Tuple, Set, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Cellular automaton predictor for combined lottery types.
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
    """Generate numbers using cellular automaton analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return list(range(min_num, min_num + required_numbers))

    # Get frequency data
    frequency_data = analyze_frequencies(past_draws)

    # Run CA analysis
    predicted_numbers = run_cellular_automaton(
        past_draws,
        frequency_data,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id'))

    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and
                 isinstance(draw.additional_numbers, list)]

    return [[int(num) for num in draw] for draw in draws]


def analyze_frequencies(past_draws: List[List[int]]) -> Dict[str, List]:
    """Analyze number frequencies in past draws."""
    all_numbers = [num for draw in past_draws for num in draw]
    number_counts = Counter(all_numbers)

    return {
        'counts': number_counts,
        'most_common': [num for num, _ in number_counts.most_common()]
    }


def run_cellular_automaton(
        past_draws: List[List[int]],
        frequency_data: Dict[str, List],
        min_num: int,
        max_num: int,
        required_numbers: int,
        rule_number: int = 30,
        iterations: int = 10
) -> List[int]:
    """Run cellular automaton analysis."""
    predicted_numbers = []
    state_length = max_num - min_num + 1

    # Convert rule number to binary
    rule = np.binary_repr(rule_number, width=8)
    rule = np.array([int(x) for x in rule], dtype=int)

    # Process each position
    for pos in range(required_numbers):
        try:
            # Initialize state
            state = create_initial_state(
                past_draws,
                pos,
                min_num,
                state_length
            )

            # Run CA iterations
            final_state = evolve_automaton(
                state,
                rule,
                iterations
            )

            # Select next number
            next_num = select_next_number(
                final_state,
                min_num,
                frequency_data,
                pos
            )

            predicted_numbers.append(next_num)

        except Exception as e:
            print(f"Error in CA analysis for position {pos}: {str(e)}")
            # Fallback to frequency-based selection
            if frequency_data['most_common']:
                predicted_numbers.append(frequency_data['most_common'][pos])

    # Ensure unique numbers
    predicted_numbers = list(dict.fromkeys(predicted_numbers))

    # Fill remaining if needed
    fill_remaining_numbers(
        predicted_numbers,
        frequency_data['most_common'],
        required_numbers
    )

    return predicted_numbers[:required_numbers]


def create_initial_state(
        past_draws: List[List[int]],
        position: int,
        min_num: int,
        state_length: int
) -> np.ndarray:
    """Create initial state for CA."""
    state = np.zeros(state_length, dtype=int)

    for draw in past_draws:
        if position < len(draw):
            num = draw[position]
            state[num - min_num] = 1

    return state


def evolve_automaton(
        state: np.ndarray,
        rule: np.ndarray,
        iterations: int
) -> np.ndarray:
    """Evolve CA state using given rule."""

    def apply_rule(left: int, center: int, right: int, rule: np.ndarray) -> int:
        index = (left << 2) | (center << 1) | right
        return rule[7 - index]

    current_state = state.copy()
    for _ in range(iterations):
        next_state = current_state.copy()
        for i in range(1, len(state) - 1):
            next_state[i] = apply_rule(
                current_state[i - 1],
                current_state[i],
                current_state[i + 1],
                rule
            )
        current_state = next_state

    return current_state


def select_next_number(
        state: np.ndarray,
        min_num: int,
        frequency_data: Dict[str, List],
        position: int
) -> int:
    """Select next number from CA state."""
    possible_numbers = [
        i + min_num
        for i, val in enumerate(state)
        if val == 1
    ]

    if possible_numbers:
        return max(possible_numbers)
    elif frequency_data['most_common']:
        return frequency_data['most_common'][position]
    else:
        return min_num + position


def fill_remaining_numbers(
        numbers: List[int],
        most_common: List[int],
        required_numbers: int
) -> None:
    """Fill remaining numbers using most common numbers."""
    for num in most_common:
        if len(numbers) >= required_numbers:
            break
        if num not in numbers:
            numbers.append(num)