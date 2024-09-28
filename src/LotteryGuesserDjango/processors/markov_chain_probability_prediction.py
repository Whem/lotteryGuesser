from collections import defaultdict
import random
from typing import List, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    transition_matrix = build_transition_matrix(past_draws, lottery_type_instance)

    predicted_numbers = []
    current_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.append(current_number)
        current_number = choose_next_number(transition_matrix, current_number, lottery_type_instance)

    return sorted(set(predicted_numbers))


def build_transition_matrix(past_draws: List[List[int]], lottery_type_instance: lg_lottery_type) -> Dict[
    int, Dict[int, float]]:
    matrix = defaultdict(lambda: defaultdict(float))
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            matrix[sorted_draw[i]][sorted_draw[i + 1]] += 1

    # Normalize probabilities
    for num in matrix:
        total = sum(matrix[num].values())
        for next_num in matrix[num]:
            matrix[num][next_num] /= total

    # Add small probability for unseen transitions
    epsilon = 0.01
    for i in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
        if i not in matrix:
            matrix[i] = defaultdict(lambda: epsilon)
        for j in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
            if j not in matrix[i]:
                matrix[i][j] = epsilon

    return matrix


def choose_next_number(matrix: Dict[int, Dict[int, float]], current: int,
                       lottery_type_instance: lg_lottery_type) -> int:
    if current not in matrix:
        return random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)

    probabilities = list(matrix[current].items())
    numbers, probs = zip(*probabilities)
    return random.choices(numbers, weights=probs)[0]