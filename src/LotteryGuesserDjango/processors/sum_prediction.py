import random
from statistics import mean
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on sum prediction.

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
    ).values_list('lottery_type_number', flat=True)

    sums = []
    for draw in past_draws:
        if isinstance(draw, list) and len(draw) == total_numbers:
            nums = [num for num in draw if isinstance(num, int)]
            if len(nums) == total_numbers:
                draw_sum = sum(nums)
                sums.append(draw_sum)

    if sums:
        avg_sum = mean(sums)
    else:
        # If no past data, generate random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Generate candidate sets matching average sum
    max_attempts = 1000
    sum_tolerance = avg_sum * 0.05  # 5% tolerance

    for _ in range(max_attempts):
        candidate_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        candidate_sum = sum(candidate_numbers)
        if abs(candidate_sum - avg_sum) <= sum_tolerance:
            selected_numbers = sorted(candidate_numbers)
            return selected_numbers

    # If no suitable candidate found, return random numbers
    selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
    selected_numbers.sort()
    return selected_numbers
