# fuzzy_logic_prediction.py
import random
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Callable
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Fuzzy logic predictor for combined lottery types.
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
    """Generate numbers using fuzzy logic analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return list(range(min_num, min_num + required_numbers))

    # Generate fuzzy membership functions
    membership_functions = create_membership_functions(min_num, max_num)

    # Get frequency data
    frequency_data = analyze_frequencies(past_draws)

    # Generate predictions using fuzzy logic
    predicted_numbers = generate_predictions(
        past_draws[-1],  # Use last draw
        membership_functions,
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


def create_membership_functions(min_num: int, max_num: int) -> Dict[str, Callable]:
    """Create fuzzy membership functions."""
    range_third = (max_num - min_num) / 3

    def low_membership(x: int) -> float:
        if x <= min_num + range_third:
            return 1.0
        elif x <= min_num + 2 * range_third:
            return (min_num + 2 * range_third - x) / range_third
        else:
            return 0.0

    def medium_membership(x: int) -> float:
        if x <= min_num + range_third or x >= min_num + 2 * range_third:
            return 0.0
        else:
            return 1.0

    def high_membership(x: int) -> float:
        if x >= min_num + 2 * range_third:
            return 1.0
        elif x >= min_num + range_third:
            return (x - (min_num + range_third)) / range_third
        else:
            return 0.0

    return {
        'LOW': low_membership,
        'MEDIUM': medium_membership,
        'HIGH': high_membership
    }


def analyze_frequencies(past_draws: List[List[int]]) -> Dict[str, List]:
    """Analyze number frequencies in past draws."""
    all_numbers = [num for draw in past_draws for num in draw]
    number_counts = Counter(all_numbers)

    return {
        'counts': number_counts,
        'most_common': [num for num, _ in number_counts.most_common()]
    }


def generate_predictions(
        last_draw: List[int],
        membership_functions: Dict[str, Callable],
        frequency_data: Dict[str, List],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions using fuzzy logic rules."""
    rules = {
        'LOW': ['LOW', 'MEDIUM'],
        'MEDIUM': ['MEDIUM', 'HIGH'],
        'HIGH': ['MEDIUM', 'HIGH']
    }

    predicted_numbers = []

    for last_num in last_draw:
        # Calculate memberships for last number
        memberships = {
            category: func(last_num)
            for category, func in membership_functions.items()
        }

        # Apply fuzzy rules
        possible_numbers = defaultdict(float)
        for category, degree in memberships.items():
            if degree > 0:
                for consequent in rules[category]:
                    for num in frequency_data['most_common']:
                        if (consequent == 'LOW' and membership_functions['LOW'](num) > 0.5 or
                                consequent == 'MEDIUM' and membership_functions['MEDIUM'](num) > 0.5 or
                                consequent == 'HIGH' and membership_functions['HIGH'](num) > 0.5):
                            possible_numbers[num] += degree * frequency_data['counts'][num]

        # Select from top numbers with randomization
        if possible_numbers:
            # Sort by score and select from top candidates
            sorted_numbers = sorted(possible_numbers.items(), key=lambda x: x[1], reverse=True)
            top_candidates_count = min(3, len(sorted_numbers))  # Top 3 candidates
            top_candidates = [num for num, _ in sorted_numbers[:top_candidates_count]]
            
            # Random selection from top candidates
            next_num = random.choice(top_candidates)
            if next_num not in predicted_numbers:
                predicted_numbers.append(next_num)

    # Fill remaining numbers if needed with randomization
    remaining_needed = required_numbers - len(predicted_numbers)
    if remaining_needed > 0:
        available_numbers = [num for num in frequency_data['most_common'] if num not in predicted_numbers]
        
        if len(available_numbers) >= remaining_needed:
            # Random selection from available numbers
            random_selection = random.sample(available_numbers, remaining_needed)
            predicted_numbers.extend(random_selection)
        else:
            # Add all available and fill with more numbers if needed
            predicted_numbers.extend(available_numbers)
            still_needed = required_numbers - len(predicted_numbers)
            
            if still_needed > 0:
                all_numbers = list(range(min_num, max_num + 1))
                unused_numbers = [num for num in all_numbers if num not in predicted_numbers]
                if unused_numbers:
                    additional_selection = random.sample(unused_numbers, min(still_needed, len(unused_numbers)))
                    predicted_numbers.extend(additional_selection)

    return predicted_numbers[:required_numbers]


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(np.random.choice(
        range(min_num, max_num + 1),
        size=required_numbers,
        replace=False
    ))