from collections import Counter
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    number_frequency = calculate_number_frequency(past_draws, lottery_type_instance)
    rarest_numbers = find_rarest_numbers(number_frequency, lottery_type_instance)

    return sorted(rarest_numbers)[:lottery_type_instance.pieces_of_draw_numbers]


def calculate_number_frequency(past_draws: List[List[int]], lottery_type_instance: lg_lottery_type) -> Counter:
    frequency = Counter()
    for draw in past_draws:
        frequency.update(draw)

    # Add unseen numbers with frequency 0
    for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
        if number not in frequency:
            frequency[number] = 0

    return frequency


def find_rarest_numbers(frequency: Counter, lottery_type_instance: lg_lottery_type) -> List[int]:
    sorted_numbers = sorted(frequency.items(), key=lambda x: x[1])
    rarest_numbers = [number for number, _ in sorted_numbers[:lottery_type_instance.pieces_of_draw_numbers]]

    return rarest_numbers