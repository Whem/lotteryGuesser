from collections import defaultdict
import random
from typing import List, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    chain_probabilities = calculate_chain_probabilities(past_draws)
    predicted_numbers = generate_number_chain(chain_probabilities, lottery_type_instance)

    return sorted(set(predicted_numbers))[:lottery_type_instance.pieces_of_draw_numbers]


def calculate_chain_probabilities(past_draws: List[List[int]]) -> Dict[int, Dict[int, float]]:
    chain_counts = defaultdict(lambda: defaultdict(int))
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            chain_counts[sorted_draw[i]][sorted_draw[i + 1]] += 1

    chain_probabilities = defaultdict(dict)
    for first, nexts in chain_counts.items():
        total = sum(nexts.values())
        for second, count in nexts.items():
            chain_probabilities[first][second] = count / total

    return chain_probabilities


def generate_number_chain(chain_probabilities: Dict[int, Dict[int, float]], lottery_type_instance: lg_lottery_type) -> \
List[int]:
    predicted_numbers = []
    current = random.choice(list(chain_probabilities.keys()))

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.append(current)
        if current in chain_probabilities:
            next_numbers = list(chain_probabilities[current].keys())
            probabilities = list(chain_probabilities[current].values())
            current = random.choices(next_numbers, weights=probabilities)[0]
        else:
            current = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)

    return predicted_numbers