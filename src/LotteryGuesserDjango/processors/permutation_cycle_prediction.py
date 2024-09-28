import random
from collections import defaultdict, Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on permutation cycle prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)
    past_draws = list(past_draws_queryset)

    # Build number appearance indices
    number_draw_indices = defaultdict(list)
    for draw_index, draw in enumerate(past_draws):
        if isinstance(draw, list):
            for num in draw:
                number_draw_indices[num].append(draw_index)

    # Predict numbers based on consistent intervals (cycles)
    predicted_numbers = []
    last_draw_index = len(past_draws) - 1

    for num, indices in number_draw_indices.items():
        if len(indices) >= 3:
            # Calculate intervals between appearances
            intervals = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            # Check if intervals are consistent
            interval_counter = Counter(intervals)
            most_common_interval, count = interval_counter.most_common(1)[0]
            # If the most common interval occurs in the majority of intervals
            if count >= len(intervals) * 0.6:
                # Predict next occurrence
                expected_next_index = indices[-1] + most_common_interval
                if expected_next_index == last_draw_index + 1:
                    predicted_numbers.append(num)

    # Ensure unique numbers and correct count
    predicted_numbers = list(set(predicted_numbers))
    num_to_select = lottery_type_instance.pieces_of_draw_numbers

    if len(predicted_numbers) < num_to_select:
        min_num = lottery_type_instance.min_number
        max_num = lottery_type_instance.max_number
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:num_to_select - len(predicted_numbers)])
    elif len(predicted_numbers) > num_to_select:
        random.shuffle(predicted_numbers)
        predicted_numbers = predicted_numbers[:num_to_select]

    predicted_numbers.sort()
    return predicted_numbers
