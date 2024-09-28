from collections import Counter
from typing import List, Dict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    divisor_counter = Counter()

    # Precompute divisors for all possible numbers
    divisor_map = {num: find_divisors(num) for num in
                   range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1)}

    for draw in past_draws:
        for number in draw:
            divisor_counter.update(divisor_map[number])

    # Get the top N most common divisors
    N = min(10, lottery_type_instance.pieces_of_draw_numbers * 2)
    common_divisors = [div for div, _ in divisor_counter.most_common(N)]

    predicted_numbers = set()
    for div in common_divisors:
        candidates = [num for num, divisors in divisor_map.items() if div in divisors]
        if candidates:
            predicted_numbers.add(random.choice(candidates))

        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    # If we don't have enough numbers, fill with random ones
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if new_number not in predicted_numbers:
            predicted_numbers.add(new_number)

    return sorted(predicted_numbers)


def find_divisors(num: int) -> List[int]:
    divisors = []
    for i in range(1, int(num ** 0.5) + 1):
        if num % i == 0:
            divisors.append(i)
            if i != num // i:
                divisors.append(num // i)
    return divisors


def analyze_divisors(past_draws: List[List[int]], top_n: int = 5) -> None:
    divisor_counter = Counter()
    for draw in past_draws:
        for number in draw:
            divisor_counter.update(find_divisors(number))

    print(f"Top {top_n} most common divisors:")
    for divisor, count in divisor_counter.most_common(top_n):
        print(f"Divisor {divisor} occurred {count} times")

    all_divisors = [div for draw in past_draws for num in draw for div in find_divisors(num)]
    print(f"\nDivisor statistics:")
    print(f"Min divisor: {min(all_divisors)}")
    print(f"Max divisor: {max(all_divisors)}")
    print(f"Mean divisor: {sum(all_divisors) / len(all_divisors):.2f}")

# Example usage of the analysis function:
# past_draws = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True))
# analyze_divisors(past_draws)