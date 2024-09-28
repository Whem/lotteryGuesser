from collections import Counter

from algorithms.models import lg_lottery_winner_number


def is_symmetric(number):
    number_str = str(number)
    return number_str == number_str[::-1]

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on the frequency of symmetrical numbers in past draws.
    """
    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    symmetry_counter = Counter()

    for draw in past_draws:
        if not isinstance(draw, list):
            continue

        for number in draw:
            if (isinstance(number, int) and
                lottery_type_instance.min_number <= number <= lottery_type_instance.max_number and
                is_symmetric(number)):
                symmetry_counter[number] += 1

    num_to_select = lottery_type_instance.pieces_of_draw_numbers
    most_common_symmetrical_numbers = symmetry_counter.most_common(num_to_select)
    selected_numbers = [num for num, count in most_common_symmetrical_numbers]

    # Fill remaining slots with random symmetrical numbers if needed
    if len(selected_numbers) < num_to_select:
        all_symmetrical_numbers = [
            num for num in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1)
            if is_symmetric(num)
        ]
        remaining_numbers = list(set(all_symmetrical_numbers) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:num_to_select - len(selected_numbers)])

    # If still not enough, fill with random numbers
    if len(selected_numbers) < num_to_select:
        all_numbers = set(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))
        used_numbers = set(selected_numbers)
        remaining_numbers = list(all_numbers - used_numbers)
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:num_to_select - len(selected_numbers)])

    selected_numbers.sort()
    return selected_numbers
