import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    range_counter = analyze_historical_ranges(past_draws)

    top_ranges = range_counter.most_common(3)
    predicted_numbers = set()

    for range_tuple, _ in top_ranges:
        range_min, range_max = range_tuple
        numbers_to_generate = max(1, lottery_type_instance.pieces_of_draw_numbers // 3)
        predicted_numbers.update(generate_numbers_in_range(range_min, range_max, numbers_to_generate))

    # Fill remaining numbers if needed
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        predicted_numbers.add(new_number)

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]


def analyze_historical_ranges(past_draws: List[List[int]]) -> Counter:
    range_counter = Counter()
    for draw in past_draws:
        sorted_draw = sorted(draw)
        range_min, range_max = sorted_draw[0], sorted_draw[-1]
        range_counter[(range_min, range_max)] += 1
    return range_counter


def generate_numbers_in_range(range_min: int, range_max: int, count: int) -> List[int]:
    return random.sample(range(range_min, range_max + 1), min(count, range_max - range_min + 1))


def calculate_range_statistics(range_counter: Counter) -> Tuple[float, float]:
    total_draws = sum(range_counter.values())
    total_range = sum((max_val - min_val) for (min_val, max_val) in range_counter.keys())
    avg_range = total_range / total_draws

    range_sizes = [max_val - min_val for (min_val, max_val) in range_counter.keys()]
    median_range = sorted(range_sizes)[len(range_sizes) // 2]

    return avg_range, median_range