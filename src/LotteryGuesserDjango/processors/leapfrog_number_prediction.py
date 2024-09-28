import random
from typing import List, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).order_by(
        '-id').values_list('lottery_type_number', flat=True)

    predicted_numbers = set()
    if past_draws.exists():
        last_draws = list(past_draws[:3])  # Consider last 3 draws
        for draw in last_draws:
            predicted_numbers.update(generate_leapfrog_numbers(draw, lottery_type_instance))

    fill_missing_numbers(predicted_numbers, lottery_type_instance)
    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]


def generate_leapfrog_numbers(draw: List[int], lottery_type_instance: lg_lottery_type) -> Set[int]:
    leapfrog_numbers = set()
    for number in draw:
        for leap in [-2, 2]:  # Consider both forward and backward leaps
            leapfrog_number = number + leap
            if lottery_type_instance.min_number <= leapfrog_number <= lottery_type_instance.max_number:
                leapfrog_numbers.add(leapfrog_number)
    return leapfrog_numbers


def fill_missing_numbers(numbers: Set[int], lottery_type_instance: lg_lottery_type) -> None:
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))