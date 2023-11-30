import random
from collections import Counter

import sympy

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    prime_counter = Counter()
    for draw in past_draws:
        for number in draw:
            if sympy.isprime(number):
                prime_counter[number] += 1

    prime_numbers = [num for num in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1) if sympy.isprime(num)]
    selected_numbers = set()
    for _ in range(lottery_type_instance.pieces_of_draw_numbers):
        if random.random() < 0.5:  # 50% chance to pick a prime number
            selected_numbers.add(random.choice(prime_numbers))
        else:
            selected_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(selected_numbers)