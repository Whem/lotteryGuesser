import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    consecutive_trends = Counter()

    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            if sorted_draw[i] + 1 == sorted_draw[i + 1]:
                consecutive_trends[(sorted_draw[i], sorted_draw[i + 1])] += 1

    predicted_numbers = set()

    # Use the top N most common trends
    N = min(3, lottery_type_instance.pieces_of_draw_numbers // 2)
    common_trends = consecutive_trends.most_common(N)

    for trend, _ in common_trends:
        predicted_numbers.update(trend)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

        # Try to extend the trend in both directions
        extend_trend(predicted_numbers, trend[0] - 1, lottery_type_instance.min_number, -1, lottery_type_instance)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break
        extend_trend(predicted_numbers, trend[1] + 1, lottery_type_instance.max_number, 1, lottery_type_instance)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    # Fill remaining numbers
    fill_remaining_numbers(predicted_numbers, lottery_type_instance)

    return sorted(list(predicted_numbers))[:lottery_type_instance.pieces_of_draw_numbers]

def extend_trend(numbers: set, start: int, limit: int, step: int, lottery_type_instance: lg_lottery_type):
    current = start
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers and current != limit:
        if lottery_type_instance.min_number <= current <= lottery_type_instance.max_number:
            numbers.add(current)
        current += step

def fill_remaining_numbers(numbers: set, lottery_type_instance: lg_lottery_type):
    all_numbers = set(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))
    available_numbers = list(all_numbers - numbers)

    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        if available_numbers:
            new_number = random.choice(available_numbers)
            numbers.add(new_number)
            available_numbers.remove(new_number)
        else:
            # Unlikely, but just in case we run out of available numbers
            break

def analyze_trends(past_draws: List[Tuple[int, ...]]) -> None:
    all_trends = Counter()
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            all_trends[sorted_draw[i + 1] - sorted_draw[i]] += 1

    print("Trend analysis:")
    for diff, count in all_trends.most_common(5):
        print(f"Difference of {diff}: occurred {count} times")