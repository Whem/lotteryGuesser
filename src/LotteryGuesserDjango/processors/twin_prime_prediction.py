import random
from algorithms.models import lg_lottery_winner_number

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True

def get_twin_primes(min_num, max_num):
    primes = [n for n in range(min_num, max_num + 1) if is_prime(n)]
    twin_primes = []
    for i in range(len(primes) - 1):
        if primes[i + 1] - primes[i] == 2:
            twin_primes.append((primes[i], primes[i + 1]))
    return twin_primes

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on twin prime prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Get all twin primes within the range
    twin_primes = get_twin_primes(min_num, max_num)

    # Flatten the twin primes into a set of numbers
    twin_prime_numbers = set()
    for pair in twin_primes:
        twin_prime_numbers.update(pair)
    twin_prime_numbers = list(twin_prime_numbers)

    # If there are not enough twin prime numbers, fill the rest with random numbers
    if len(twin_prime_numbers) >= total_numbers:
        selected_numbers = random.sample(twin_prime_numbers, total_numbers)
    else:
        selected_numbers = twin_prime_numbers.copy()
        remaining_slots = total_numbers - len(selected_numbers)
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:remaining_slots])

    # Ensure unique numbers and correct count
    selected_numbers = list(set(selected_numbers))
    if len(selected_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])
    elif len(selected_numbers) > total_numbers:
        selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the numbers
    selected_numbers.sort()
    return selected_numbers
