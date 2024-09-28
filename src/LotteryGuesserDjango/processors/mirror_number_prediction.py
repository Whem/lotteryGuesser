from typing import List, Set
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    mirror_numbers = generate_mirror_numbers(past_draws, lottery_type_instance)

    predicted_numbers = set(mirror_numbers)
    fill_missing_numbers(predicted_numbers, lottery_type_instance)

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]


def generate_mirror_numbers(past_draws: List[List[int]], lottery_type_instance: lg_lottery_type) -> Set[int]:
    mirror_numbers = set()
    for draw in past_draws:
        for number in draw:
            mirror = get_mirror_number(number, lottery_type_instance)
            if mirror:
                mirror_numbers.add(mirror)
    return mirror_numbers


def get_mirror_number(number: int, lottery_type_instance: lg_lottery_type) -> int:
    mirror = int(str(number)[::-1])
    if lottery_type_instance.min_number <= mirror <= lottery_type_instance.max_number:
        return mirror
    return 0


def fill_missing_numbers(numbers: Set[int], lottery_type_instance: lg_lottery_type) -> None:
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))