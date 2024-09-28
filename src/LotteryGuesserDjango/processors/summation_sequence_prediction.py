import random
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on summation sequence prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve sums of past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    sum_sequence = []
    for draw in past_draws:
        if isinstance(draw, list) and len(draw) == total_numbers:
            nums = [num for num in draw if isinstance(num, int)]
            if len(nums) == total_numbers:
                draw_sum = sum(nums)
                sum_sequence.append(draw_sum)

    if len(sum_sequence) >= 3:
        # Calculate differences between consecutive sums
        sum_differences = [sum_sequence[i+1] - sum_sequence[i] for i in range(len(sum_sequence)-1)]
        # Calculate average difference (assumes linear trend)
        avg_difference = sum(sum_differences) / len(sum_differences)
        # Predict the next sum in the sequence
        predicted_next_sum = sum_sequence[-1] + avg_difference

        # Generate candidate sets matching the predicted sum
        max_attempts = 1000
        sum_tolerance = predicted_next_sum * 0.05  # 5% tolerance

        for _ in range(max_attempts):
            candidate_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
            candidate_sum = sum(candidate_numbers)
            if abs(candidate_sum - predicted_next_sum) <= sum_tolerance:
                selected_numbers = sorted(candidate_numbers)
                return selected_numbers
    else:
        # If not enough data, generate random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # If no suitable candidate found, return random numbers
    selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
    selected_numbers.sort()
    return selected_numbers
