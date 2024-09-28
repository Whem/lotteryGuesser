import random
from statistics import mean, stdev
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on standard deviation-based prediction.

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

    # Calculate standard deviations and means of past draws
    stdevs = []
    means = []
    for draw in past_draws:
        if isinstance(draw, list) and len(draw) == total_numbers:
            nums = [num for num in draw if isinstance(num, int)]
            if len(nums) == total_numbers:
                draw_mean = mean(nums)
                draw_stdev = stdev(nums)
                means.append(draw_mean)
                stdevs.append(draw_stdev)

    if stdevs and means:
        avg_stdev = mean(stdevs)
        avg_mean = mean(means)
    else:
        # If no past data, return random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Generate candidate sets matching average standard deviation and mean
    max_attempts = 1000
    stdev_tolerance = avg_stdev * 0.1  # 10% tolerance
    mean_tolerance = (max_num - min_num) * 0.05  # 5% of range

    for _ in range(max_attempts):
        candidate_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        candidate_mean = mean(candidate_numbers)
        candidate_stdev = stdev(candidate_numbers)
        if (abs(candidate_stdev - avg_stdev) <= stdev_tolerance and
            abs(candidate_mean - avg_mean) <= mean_tolerance):
            selected_numbers = sorted(candidate_numbers)
            return selected_numbers

    # If no suitable candidate found, return random numbers
    selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
    selected_numbers.sort()
    return selected_numbers
