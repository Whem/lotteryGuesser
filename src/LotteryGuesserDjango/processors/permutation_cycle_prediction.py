# permutation_cycle_prediction.py

import random
from collections import defaultdict, Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on permutation cycle prediction.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        num_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            num_numbers=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_numbers(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        num_numbers: int,
        numbers_field: str
) -> List[int]:
    """
    Generates a set of lottery numbers based on permutation cycle prediction.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(numbers_field, flat=True)
    past_draws = list(past_draws_queryset)

    # Build number appearance indices
    number_draw_indices = defaultdict(list)
    for draw_index, draw in enumerate(past_draws):
        if isinstance(draw, list):
            for num in draw:
                number_draw_indices[int(num)].append(draw_index)

    # Predict numbers based on consistent intervals (cycles)
    predicted_numbers = []
    last_draw_index = len(past_draws) - 1

    for num, indices in number_draw_indices.items():
        if len(indices) >= 3:
            # Calculate intervals between appearances
            intervals = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
            # Check if intervals are consistent
            interval_counter = Counter(intervals)
            most_common_interval, count = interval_counter.most_common(1)[0]
            # If the most common interval occurs in the majority of intervals
            if count >= len(intervals) * 0.6:
                # Predict next occurrence
                expected_next_index = indices[-1] + most_common_interval
                if expected_next_index == last_draw_index + 1:
                    predicted_numbers.append(int(num))

    # Ensure unique numbers and correct count
    predicted_numbers = list(set(predicted_numbers))

    if len(predicted_numbers) < num_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:num_numbers - len(predicted_numbers)])
    elif len(predicted_numbers) > num_numbers:
        random.shuffle(predicted_numbers)
        predicted_numbers = predicted_numbers[:num_numbers]

    # Convert to standard Python int and sort
    predicted_numbers = [int(num) for num in predicted_numbers]
    predicted_numbers.sort()
    return predicted_numbers
