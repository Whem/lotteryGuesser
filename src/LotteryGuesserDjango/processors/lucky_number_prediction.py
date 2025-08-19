# lucky_number_prediction.py
from collections import Counter
from typing import List, Tuple, Set, Dict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Generate lottery numbers using luckiness analysis."""
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

    return sorted(main_numbers), sorted(additional_numbers)


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using luckiness analysis."""
    # Get historical data
    past_numbers = get_historical_numbers(
        lottery_type_instance,
        min_num,
        max_num,
        is_main
    )

    if not past_numbers:
        return random_selection(min_num, max_num, required_numbers)

    # Calculate luck scores
    luck_scores = calculate_luck_scores(
        past_numbers,
        min_num,
        max_num
    )

    # Generate predictions
    lucky_numbers = generate_predictions(
        luck_scores,
        min_num,
        max_num,
        required_numbers
    )

    return lucky_numbers


def get_historical_numbers(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        is_main: bool
) -> List[int]:
    """Get historical lottery numbers."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    past_numbers = []
    for draw in past_draws:
        if is_main and isinstance(draw.lottery_type_number, list):
            numbers = [
                num for num in draw.lottery_type_number
                if min_num <= num <= max_num
            ]
            past_numbers.extend(numbers)
        elif not is_main and hasattr(draw, 'additional_numbers'):
            numbers = [
                num for num in draw.additional_numbers
                if min_num <= num <= max_num
            ]
            past_numbers.extend(numbers)

    return past_numbers


def calculate_luck_scores(
        past_numbers: List[int],
        min_num: int,
        max_num: int
) -> Dict[int, float]:
    """Calculate luckiness scores for numbers."""
    # Count frequencies
    frequency_counter = Counter(past_numbers)

    # Add never drawn numbers
    all_numbers = set(range(min_num, max_num + 1))
    never_drawn = all_numbers - set(frequency_counter.keys())
    for num in never_drawn:
        frequency_counter[num] = 0

    # Calculate scores
    total_draws = len(past_numbers) / max(
        1,
        max(len(draw) for draw in [past_numbers[i:i + 5] for i in range(0, len(past_numbers), 5)])
    )

    return {
        num: count / total_draws
        for num, count in frequency_counter.items()
    }


def generate_predictions(
        luck_scores: Dict[int, float],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions based on luck scores with randomization."""
    # Sort by luck score
    sorted_numbers = sorted(
        luck_scores.keys(),
        key=luck_scores.get,
        reverse=True
    )

    # Ensure enough numbers
    if len(sorted_numbers) < required_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(sorted_numbers)
        sorted_numbers.extend(list(remaining))

    # Randomized selection from top candidates
    top_candidates_count = min(required_numbers * 3, len(sorted_numbers))
    top_candidates = sorted_numbers[:top_candidates_count]
    
    # Mix guaranteed top picks with random selection
    guaranteed_count = max(1, int(required_numbers * 0.7))  # 70% guaranteed
    guaranteed_numbers = top_candidates[:guaranteed_count]
    
    # Random selection for remaining slots
    remaining_needed = required_numbers - len(guaranteed_numbers)
    if remaining_needed > 0:
        remaining_candidates = top_candidates[guaranteed_count:]
        if len(remaining_candidates) >= remaining_needed:
            random_selection = random.sample(remaining_candidates, remaining_needed)
        else:
            random_selection = remaining_candidates[:]
            # Fill with additional random numbers if needed
            all_numbers = set(range(min_num, max_num + 1))
            unused_numbers = list(all_numbers - set(guaranteed_numbers) - set(random_selection))
            if unused_numbers:
                additional_needed = remaining_needed - len(random_selection)
                random_selection.extend(random.sample(unused_numbers, min(additional_needed, len(unused_numbers))))
        
        lucky_numbers = guaranteed_numbers + random_selection[:remaining_needed]
    else:
        lucky_numbers = guaranteed_numbers

    # Convert to standard Python int and ensure correct count
    lucky_numbers = [int(num) for num in lucky_numbers[:required_numbers]]

    return lucky_numbers


def random_selection(min_num: int, max_num: int, count: int) -> List[int]:
    """Generate random number selection."""
    all_numbers = set(range(min_num, max_num + 1))
    return sorted(random.sample(all_numbers, count))