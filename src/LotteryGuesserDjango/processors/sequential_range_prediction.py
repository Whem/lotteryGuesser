# sequential_range_prediction.py
import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on sequential range prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance=lottery_type_instance,
        number_field='lottery_type_number',
        min_num=lottery_type_instance.min_number,
        max_num=lottery_type_instance.max_number,
        total_numbers=lottery_type_instance.pieces_of_draw_numbers
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=lottery_type_instance.additional_min_number,
            max_num=lottery_type_instance.additional_max_number,
            total_numbers=lottery_type_instance.additional_numbers_count
        )

    return main_numbers, additional_numbers


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers based on sequential range prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers
                    ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(number_field, flat=True)

    past_draws = [
        sorted(set(draw)) for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    # Count frequency of sequential ranges
    sequence_counter = Counter()
    for draw in past_draws:
        sequences = []
        if not draw:
            continue
        seq_start = draw[0]
        seq_length = 1
        for i in range(1, len(draw)):
            if draw[i] == draw[i - 1] + 1:
                seq_length += 1
            else:
                if seq_length >= 2:
                    sequences.append((seq_start, seq_length))
                seq_start = draw[i]
                seq_length = 1
        if seq_length >= 2:
            sequences.append((seq_start, seq_length))
        for seq in sequences:
            sequence_counter[seq] += 1

    # Find the most common sequential ranges
    selected_numbers = []
    if sequence_counter:
        # Sort sequences by frequency and then by sequence length (descending)
        most_common_sequences = sorted(
            sequence_counter.items(),
            key=lambda x: (-x[1], -x[0][1])
        )
        for (seq_start, seq_length), _ in most_common_sequences:
            seq_numbers = [seq_start + i for i in range(seq_length)]
            selected_numbers.extend(seq_numbers)
            if len(selected_numbers) >= total_numbers:
                break
        selected_numbers = selected_numbers[:total_numbers]
    else:
        # If no sequential ranges found, select random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)

    # Ensure numbers are within range and unique
    selected_numbers = [num for num in selected_numbers if min_num <= num <= max_num]
    selected_numbers = list(set(selected_numbers))

    # If not enough numbers, fill with random numbers
    if len(selected_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    # Ensure exactly total_numbers are selected
    selected_numbers = selected_numbers[:total_numbers]
    selected_numbers.sort()
    return selected_numbers
