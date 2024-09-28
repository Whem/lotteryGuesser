from collections import Counter
from typing import List, Set
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    number_counter = Counter(num for draw in past_draws for num in draw)

    hot_cold_ratio = 0.6
    hot_count = int(lottery_type_instance.pieces_of_draw_numbers * hot_cold_ratio)
    cold_count = lottery_type_instance.pieces_of_draw_numbers - hot_count

    hot_numbers = set(num for num, _ in number_counter.most_common(hot_count))
    cold_numbers = set(num for num, _ in number_counter.most_common()[:-cold_count-1:-1])

    selected_numbers = hot_numbers.union(cold_numbers)

    fill_missing_numbers(selected_numbers, lottery_type_instance)

    return sorted(selected_numbers)

def fill_missing_numbers(numbers: Set[int], lottery_type_instance: lg_lottery_type) -> None:
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))