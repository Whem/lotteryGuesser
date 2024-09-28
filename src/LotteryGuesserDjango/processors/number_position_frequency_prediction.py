from collections import defaultdict
import random
from typing import List, Dict, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    position_frequencies = calculate_position_frequencies(past_draws, lottery_type_instance)
    predicted_numbers = generate_numbers_from_frequencies(position_frequencies, lottery_type_instance)

    return sorted(predicted_numbers)


def calculate_position_frequencies(past_draws: List[List[int]], lottery_type_instance: lg_lottery_type) -> Dict[
    int, Dict[int, int]]:
    frequencies = defaultdict(lambda: defaultdict(int))

    for draw in past_draws:
        sorted_draw = sorted(draw)
        for position, number in enumerate(sorted_draw):
            frequencies[position][number] += 1

    return frequencies


def generate_numbers_from_frequencies(frequencies: Dict[int, Dict[int, int]], lottery_type_instance: lg_lottery_type) -> \
List[int]:
    predicted_numbers = []

    for position in range(lottery_type_instance.pieces_of_draw_numbers):
        position_freq = frequencies[position]
        if position_freq:
            number = max(position_freq, key=position_freq.get)
            predicted_numbers.append(number)
        else:
            number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
            predicted_numbers.append(number)

    # Ensure uniqueness and correct count
    unique_numbers = list(set(predicted_numbers))
    while len(unique_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if new_number not in unique_numbers:
            unique_numbers.append(new_number)

    return unique_numbers[:lottery_type_instance.pieces_of_draw_numbers]