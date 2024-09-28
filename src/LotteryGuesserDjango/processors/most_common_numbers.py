from collections import Counter
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    number_counter = Counter(num for draw in past_draws for num in draw)

    all_numbers = set(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))
    never_drawn = all_numbers - set(number_counter.keys())

    for num in never_drawn:
        number_counter[num] = 0

    most_common = number_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)
    predicted_numbers = [num for num, _ in most_common]

    return predicted_numbers