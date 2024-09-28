import random
from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on sequential range prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Count frequency of sequential ranges
    sequence_counter = Counter()
    for draw in past_draws:
        if isinstance(draw, list):
            sorted_draw = sorted(set(draw))
            sequences = []
            seq_start = sorted_draw[0]
            seq_length = 1
            for i in range(1, len(sorted_draw)):
                if sorted_draw[i] == sorted_draw[i - 1] + 1:
                    seq_length += 1
                else:
                    if seq_length >= 2:
                        sequences.append((seq_start, seq_length))
                    seq_start = sorted_draw[i]
                    seq_length = 1
            if seq_length >= 2:
                sequences.append((seq_start, seq_length))
            for seq in sequences:
                sequence_counter[seq] += 1

    # Find the most common sequential ranges
    selected_numbers = []
    if sequence_counter:
        # Sort sequences by frequency and length
        most_common_sequences = sorted(sequence_counter.items(), key=lambda x: (-x[1], -x[0][1]))
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

    # Sort and return the numbers
    selected_numbers = selected_numbers[:total_numbers]
    selected_numbers.sort()
    return selected_numbers
