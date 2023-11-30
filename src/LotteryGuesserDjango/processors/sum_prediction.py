#sum_prediction.py
import random
from collections import Counter
from itertools import combinations

from algorithms.models import lg_lottery_winner_number


def find_combinations_that_sum_to(target_sum, min_number, max_number, number_count):
    possible_numbers = list(range(min_number, max_number + 1))
    valid_combinations = [combo for combo in combinations(possible_numbers, number_count) if sum(combo) == target_sum]
    return random.choice(valid_combinations) if valid_combinations else None


def predict_next_sum(lottery_type):
    queryset = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).values_list('sum', flat=True))


    sum_counter = Counter(queryset)

    # Find the most common sum that follows the last sum
    last_sum = queryset[-1] if queryset else None
    next_sum = max(sum_counter, key=lambda x: sum_counter[x] if x != last_sum else -1)

    return next_sum


def get_numbers(lottery_type):
    next_sum = predict_next_sum(lottery_type)
    if next_sum is not None:
        return find_combinations_that_sum_to(next_sum, lottery_type.min_number, lottery_type.max_number,
                                             lottery_type.pieces_of_draw_numbers)
    else:
        return None