import random
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on random walk prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve the most recent winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id').values_list('lottery_type_number', flat=True)

    if past_draws.exists():
        # Start from the most recent draw
        last_draw = past_draws.first()
        if isinstance(last_draw, list) and len(last_draw) == total_numbers:
            current_numbers = sorted(last_draw)
        else:
            # If last draw is not usable, start from random numbers
            current_numbers = sorted(random.sample(range(min_num, max_num + 1), total_numbers))
    else:
        # If no past draws, start from random numbers
        current_numbers = sorted(random.sample(range(min_num, max_num + 1), total_numbers))

    # Perform a random walk
    predicted_numbers = []
    for num in current_numbers:
        # Decide to move up, down, or stay the same
        move = random.choice([-1, 0, 1])
        new_num = num + move

        # Ensure the new number is within bounds
        if new_num < min_num:
            new_num = min_num
        elif new_num > max_num:
            new_num = max_num

        predicted_numbers.append(new_num)

    # Ensure unique numbers
    predicted_numbers = list(set(predicted_numbers))

    # If not enough numbers, fill with random numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    elif len(predicted_numbers) > total_numbers:
        predicted_numbers = predicted_numbers[:total_numbers]

    # Sort and return the selected numbers
    predicted_numbers.sort()
    return predicted_numbers
