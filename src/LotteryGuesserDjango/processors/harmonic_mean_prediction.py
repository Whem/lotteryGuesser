import random
from typing import List
from statistics import harmonic_mean
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    all_numbers = [number for draw in past_draws for number in draw if number != 0]

    if not all_numbers:
        return random_selection(lottery_type_instance)

    hm = harmonic_mean(all_numbers)
    std_dev = calculate_standard_deviation(all_numbers)

    predicted_numbers = set()
    attempts = 0
    max_attempts = lottery_type_instance.pieces_of_draw_numbers * 10

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers and attempts < max_attempts:
        number = generate_number(hm, std_dev, lottery_type_instance)
        if number:
            predicted_numbers.add(number)
        attempts += 1

    if len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.update(random_selection(lottery_type_instance, lottery_type_instance.pieces_of_draw_numbers - len(predicted_numbers)))

    return sorted(predicted_numbers)

def generate_number(hm: float, std_dev: float, lottery_type_instance: lg_lottery_type) -> int:
    number = int(hm + random.gauss(0, std_dev))
    if lottery_type_instance.min_number <= number <= lottery_type_instance.max_number:
        return number
    return 0

def calculate_standard_deviation(numbers: List[int]) -> float:
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return variance ** 0.5

def random_selection(lottery_type_instance: lg_lottery_type, count: int = None) -> List[int]:
    if count is None:
        count = lottery_type_instance.pieces_of_draw_numbers
    return random.sample(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1), count)