from collections import Counter
from typing import List, Dict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    modulo_patterns = analyze_modulo_patterns(past_draws, lottery_type_instance)
    predicted_numbers = generate_numbers_from_patterns(modulo_patterns, lottery_type_instance)

    # Ha túl sok számot generáltunk, véletlenszerűen eltávolítunk néhányat
    while len(predicted_numbers) > lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.remove(random.choice(list(predicted_numbers)))

    # Ha nem elég számot generáltunk, feltöltjük random számokkal
    fill_missing_numbers(predicted_numbers, lottery_type_instance)

    return sorted(list(predicted_numbers))

def analyze_modulo_patterns(past_draws: List[List[int]], lottery_type_instance: lg_lottery_type) -> Dict[int, Counter]:
    patterns = {i: Counter() for i in range(2, 11)}  # Analyze patterns for modulo 2 to 10
    for draw in past_draws:
        for number in draw:
            for modulo in patterns:
                patterns[modulo][number % modulo] += 1
    return patterns

def generate_numbers_from_patterns(patterns: Dict[int, Counter], lottery_type_instance: lg_lottery_type) -> set:
    predicted_numbers = set()
    for modulo, counter in patterns.items():
        most_common = counter.most_common(2)
        for remainder, _ in most_common:
            candidates = [num for num in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1)
                          if num % modulo == remainder]
            if candidates:
                new_number = random.choice(candidates)
                predicted_numbers.add(new_number)
                if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                    return predicted_numbers
    return predicted_numbers

def fill_missing_numbers(numbers: set, lottery_type_instance: lg_lottery_type) -> None:
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if new_number not in numbers:
            numbers.add(new_number)