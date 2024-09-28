import random
from collections import Counter
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    cumulative_sums = [sum(draw) for draw in past_draws]
    sum_counter = Counter(cumulative_sums)

    # Get the top 3 most common sums
    common_sums = sum_counter.most_common(3)

    for target_sum, _ in common_sums:
        predicted_numbers = find_numbers_with_sum(target_sum, lottery_type_instance)
        if predicted_numbers:
            return sorted(predicted_numbers)

    # If we couldn't find a matching set for any of the common sums, return random numbers
    return generate_random_numbers(lottery_type_instance)


def find_numbers_with_sum(target_sum: int, lottery_type_instance: lg_lottery_type, max_attempts: int = 1000) -> List[
    int]:
    for _ in range(max_attempts):
        numbers = set()
        current_sum = 0
        while current_sum < target_sum and len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
            remaining = target_sum - current_sum
            max_possible = min(remaining, lottery_type_instance.max_number)
            new_number = random.randint(lottery_type_instance.min_number, max_possible)
            if new_number not in numbers:
                numbers.add(new_number)
                current_sum += new_number

        if current_sum == target_sum and len(numbers) == lottery_type_instance.pieces_of_draw_numbers:
            return list(numbers)

    return []  # If we couldn't find a matching set


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    return random.sample(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
                         lottery_type_instance.pieces_of_draw_numbers)