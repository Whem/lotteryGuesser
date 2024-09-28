from collections import Counter
from typing import List, Tuple
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True))
    number_correlations = calculate_correlations(past_draws)

    predicted_numbers = set()
    for pair, _ in number_correlations.most_common():
        predicted_numbers.update(pair)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    # Ha túl sok számot kaptunk, véletlenszerűen eltávolítunk néhányat
    while len(predicted_numbers) > lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.remove(random.choice(list(predicted_numbers)))

    # Ha nem elég számot kaptunk, feltöltjük random számokkal
    fill_missing_numbers(predicted_numbers, lottery_type_instance)

    return sorted(list(predicted_numbers))[:lottery_type_instance.pieces_of_draw_numbers]

def calculate_correlations(past_draws: List[Tuple[int, ...]]) -> Counter:
    correlations = Counter()
    for i in range(len(past_draws) - 1):
        for number in past_draws[i]:
            for next_number in past_draws[i + 1]:
                correlations[(number, next_number)] += 1
    return correlations

def fill_missing_numbers(numbers: set, lottery_type_instance: lg_lottery_type) -> None:
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if new_number not in numbers:
            numbers.add(new_number)