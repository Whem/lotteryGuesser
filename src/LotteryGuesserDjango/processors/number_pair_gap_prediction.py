from collections import defaultdict
import random
from typing import List, Dict, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    pair_gaps = calculate_pair_gaps(past_draws)
    predicted_numbers = generate_numbers_from_gaps(pair_gaps, lottery_type_instance)

    return sorted(set(predicted_numbers))[:lottery_type_instance.pieces_of_draw_numbers]


def calculate_pair_gaps(past_draws: List[List[int]]) -> Dict[Tuple[int, int], List[int]]:
    pair_gaps = defaultdict(list)
    pair_last_seen = {}

    for draw_index, draw in enumerate(past_draws):
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                pair = tuple(sorted([draw[i], draw[j]]))
                if pair in pair_last_seen:
                    gap = draw_index - pair_last_seen[pair]
                    pair_gaps[pair].append(gap)
                pair_last_seen[pair] = draw_index

    return pair_gaps


def generate_numbers_from_gaps(pair_gaps: Dict[Tuple[int, int], List[int]], lottery_type_instance: lg_lottery_type) -> \
List[int]:
    avg_gaps = {pair: sum(gaps) / len(gaps) for pair, gaps in pair_gaps.items()}
    sorted_pairs = sorted(avg_gaps.items(), key=lambda x: x[1])

    predicted_numbers = []
    for pair, _ in sorted_pairs:
        predicted_numbers.extend(pair)
        if len(set(predicted_numbers)) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    while len(set(predicted_numbers)) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        predicted_numbers.append(new_number)

    return predicted_numbers