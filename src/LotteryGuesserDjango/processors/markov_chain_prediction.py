# markov_chain_prediction.py
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Markov chain predictor for combined lottery types."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

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
    """Generate numbers using Markov chain analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return list(range(min_num, min_num + required_numbers))

    # Build Markov chain
    transition_probs = build_transition_matrix(past_draws)

    # Generate predictions
    predicted_numbers = generate_predictions(
        transition_probs,
        past_draws[-1],
        min_num,
        max_num,
        required_numbers
    )

    return predicted_numbers


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
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


def build_transition_matrix(past_draws: List[List[int]]) -> Dict[int, Dict[int, float]]:
    """Build Markov chain transition matrix."""
    transition_counts = defaultdict(Counter)

    for draw in past_draws:
        for i in range(len(draw) - 1):
            current_num = draw[i]
            next_num = draw[i + 1]
            transition_counts[current_num][next_num] += 1

    # Calculate probabilities
    transition_probs = {}
    for current_num, counter in transition_counts.items():
        total = sum(counter.values())
        transition_probs[current_num] = {
            num: count / total
            for num, count in counter.items()
        }

    return transition_probs


def generate_predictions(
        transition_probs: Dict[int, Dict[int, float]],
        last_draw: List[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions using Markov chain."""
    predicted_numbers = set()

    # Use transition probabilities
    for num in last_draw:
        if num in transition_probs:
            next_num = max(
                transition_probs[num].items(),
                key=lambda x: x[1]
            )[0]
            if min_num <= next_num <= max_num:
                predicted_numbers.add(next_num)

    # Fill with most common transitions if needed
    if len(predicted_numbers) < required_numbers:
        common_transitions = get_common_transitions(transition_probs)
        for num in common_transitions:
            if min_num <= num <= max_num:
                predicted_numbers.add(num)
                if len(predicted_numbers) >= required_numbers:
                    break

    # Fill remaining deterministically if needed
    fill_remaining_numbers(
        predicted_numbers,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(list(predicted_numbers)[:required_numbers])


def get_common_transitions(transition_probs: Dict[int, Dict[int, float]]) -> List[int]:
    """Get most common transition targets."""
    target_counts = Counter()

    for transitions in transition_probs.values():
        for num, prob in transitions.items():
            target_counts[num] += prob

    return [num for num, _ in target_counts.most_common()]


def fill_remaining_numbers(
        numbers: Set[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> None:
    """Fill missing numbers deterministically."""
    for num in range(min_num, max_num + 1):
        if len(numbers) >= required_numbers:
            break
        if num not in numbers:
            numbers.add(num)