from collections import defaultdict
import random
from typing import List, Dict, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    pair_affinities = calculate_pair_affinities(past_draws)
    predicted_numbers = generate_numbers_from_affinities(pair_affinities, lottery_type_instance)

    return sorted(set(predicted_numbers))[:lottery_type_instance.pieces_of_draw_numbers]


def calculate_pair_affinities(past_draws: List[List[int]]) -> Dict[Tuple[int, int], int]:
    pair_counts = defaultdict(int)
    for draw in past_draws:
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                pair = tuple(sorted([draw[i], draw[j]]))
                pair_counts[pair] += 1
    return pair_counts


def generate_numbers_from_affinities(pair_affinities: Dict[Tuple[int, int], int],
                                     lottery_type_instance: lg_lottery_type) -> List[int]:
    predicted_numbers = []
    sorted_pairs = sorted(pair_affinities.items(), key=lambda x: x[1], reverse=True)

    for pair, _ in sorted_pairs:
        predicted_numbers.extend(pair)
        if len(set(predicted_numbers)) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    while len(set(predicted_numbers)) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        predicted_numbers.append(new_number)

    return predicted_numbers