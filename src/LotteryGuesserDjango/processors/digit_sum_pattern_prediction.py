from collections import Counter
from typing import List, Dict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    digit_sum_counter = Counter()

    for draw in past_draws:
        for number in draw:
            digit_sum_counter[digit_sum(number)] += 1

    # Get the top 3 most common digit sums
    common_digit_sums = [sum for sum, _ in digit_sum_counter.most_common(3)]

    # Precompute digit sums for all possible numbers
    digit_sum_map = {num: digit_sum(num) for num in
                     range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1)}

    predicted_numbers = set()
    for target_sum in common_digit_sums:
        candidates = [num for num, sum in digit_sum_map.items() if sum == target_sum]
        predicted_numbers.update(random.sample(candidates, min(lottery_type_instance.pieces_of_draw_numbers - len(
            predicted_numbers), len(candidates))))

        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    # If we don't have enough numbers, fill with random ones
    if len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        remaining = lottery_type_instance.pieces_of_draw_numbers - len(predicted_numbers)
        additional_numbers = random.sample(
            range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1), remaining)
        predicted_numbers.update(additional_numbers)

    return sorted(predicted_numbers)


def digit_sum(number: int) -> int:
    return sum(int(digit) for digit in str(number))


def analyze_digit_sums(past_draws: List[List[int]], top_n: int = 5) -> None:
    digit_sum_counter = Counter()
    for draw in past_draws:
        for number in draw:
            digit_sum_counter[digit_sum(number)] += 1

    print(f"Top {top_n} most common digit sums:")
    for sum, count in digit_sum_counter.most_common(top_n):
        print(f"Digit sum {sum} occurred {count} times")

    all_sums = [digit_sum(num) for draw in past_draws for num in draw]
    print(f"\nDigit sum statistics:")
    print(f"Min sum: {min(all_sums)}")
    print(f"Max sum: {max(all_sums)}")
    print(f"Mean sum: {sum(all_sums) / len(all_sums):.2f}")

# Example usage of the analysis function:
# past_draws = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True))
# analyze_digit_sums(past_draws)