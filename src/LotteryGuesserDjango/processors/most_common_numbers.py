# most_common_numbers.py

import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate the most common numbers from past draws for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_most_common_numbers(
        lottery_type_instance,
        min_number=int(lottery_type_instance.min_number),
        max_number=int(lottery_type_instance.max_number),
        pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_most_common_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.additional_min_number),
            max_number=int(lottery_type_instance.additional_max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_most_common_numbers(
        lottery_type_instance: lg_lottery_type,
        min_number: int,
        max_number: int,
        pieces_of_draw_numbers: int,
        numbers_field: str
) -> List[int]:
    """
    Generate the most common numbers from past draws.
    """
    # Retrieve past draws
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(numbers_field, flat=True)

    # Flatten the list of past numbers
    number_counter = Counter(num for draw in past_draws if isinstance(draw, list) for num in draw)

    # Ensure all numbers in the range are included in the counter
    all_numbers = set(range(min_number, max_number + 1))
    never_drawn = all_numbers - set(number_counter.keys())

    for num in never_drawn:
        number_counter[num] = 0

    # Get a broader selection of common numbers for randomization
    top_candidates_count = min(pieces_of_draw_numbers * 3, len(number_counter))
    top_candidates = number_counter.most_common(top_candidates_count)
    
    # Mix of guaranteed top picks and random selection
    guaranteed_count = max(1, int(pieces_of_draw_numbers * 0.6))  # 60% guaranteed top
    guaranteed_numbers = [int(num) for num, _ in top_candidates[:guaranteed_count]]
    
    # Random selection from remaining candidates
    remaining_needed = pieces_of_draw_numbers - len(guaranteed_numbers)
    if remaining_needed > 0:
        remaining_candidates = [int(num) for num, _ in top_candidates[guaranteed_count:]]
        if len(remaining_candidates) >= remaining_needed:
            random_selection = random.sample(remaining_candidates, remaining_needed)
        else:
            # If not enough candidates, fill with all remaining and add random numbers
            random_selection = remaining_candidates[:]
            all_numbers = list(range(min_number, max_number + 1))
            unused_numbers = [n for n in all_numbers if n not in guaranteed_numbers + random_selection]
            if unused_numbers:
                additional_needed = remaining_needed - len(random_selection)
                random_selection.extend(random.sample(unused_numbers, min(additional_needed, len(unused_numbers))))
        
        predicted_numbers = guaranteed_numbers + random_selection[:remaining_needed]
    else:
        predicted_numbers = guaranteed_numbers
    
    # Ensure we have exactly the right number
    predicted_numbers = predicted_numbers[:pieces_of_draw_numbers]
    
    return predicted_numbers
