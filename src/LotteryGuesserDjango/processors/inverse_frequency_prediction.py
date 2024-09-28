from collections import Counter
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    frequency_counter = Counter(num for draw in past_draws for num in draw)

    total_draws = len(past_draws)
    inverse_frequency = {num: total_draws / (freq + 1) for num, freq in frequency_counter.items()}

    all_numbers = set(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))
    never_drawn = all_numbers - set(frequency_counter.keys())

    for num in never_drawn:
        inverse_frequency[num] = total_draws

    predicted_numbers = sorted(inverse_frequency, key=inverse_frequency.get, reverse=True)[
                        :lottery_type_instance.pieces_of_draw_numbers]

    return predicted_numbers