from collections import Counter

import sympy

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    prime_gap_counter = Counter()

    for draw in past_draws:
        primes = [num for num in draw if sympy.isprime(num)]
        for i in range(len(primes) - 1):
            gap = primes[i + 1] - primes[i]
            prime_gap_counter[gap] += 1

    most_common_prime_gap = prime_gap_counter.most_common(1)[0][0]
    predicted_numbers = set()
    for num in range(lottery_type_instance.min_number, lottery_type_instance.max_number):
        if sympy.isprime(num) and sympy.isprime(num + most_common_prime_gap):
            predicted_numbers.add(num)
            predicted_numbers.add(num + most_common_prime_gap)
            if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                break

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]
