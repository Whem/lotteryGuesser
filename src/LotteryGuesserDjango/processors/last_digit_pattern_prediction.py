from collections import Counter
from typing import List, Set
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    last_digit_counter = Counter(number % 10 for draw in past_draws for number in draw)

    most_common_last_digits = [digit for digit, _ in last_digit_counter.most_common()]
    predicted_numbers = generate_numbers_from_digits(most_common_last_digits, lottery_type_instance)

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]


def generate_numbers_from_digits(digits: List[int], lottery_type_instance: lg_lottery_type) -> Set[int]:
    numbers = set()
    for digit in digits:
        candidates = [num for num in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1) if
                      num % 10 == digit]
        if candidates:
            numbers.add(random.choice(candidates))
        if len(numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return numbers