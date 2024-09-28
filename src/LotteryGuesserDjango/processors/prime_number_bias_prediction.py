import random
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers with a bias towards prime numbers.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Helper function to check if a number is prime
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_needed = lottery_type_instance.pieces_of_draw_numbers

    # Generate a list of all numbers in the range
    all_numbers = list(range(min_num, max_num + 1))
    # Separate primes and non-primes
    prime_numbers = [n for n in all_numbers if is_prime(n)]
    non_prime_numbers = [n for n in all_numbers if not is_prime(n)]

    # Determine the proportion of primes in past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    total_numbers = 0
    prime_count = 0
    for draw in past_draws:
        if not isinstance(draw, list):
            continue
        for number in draw:
            total_numbers += 1
            if is_prime(number):
                prime_count += 1

    if total_numbers > 0:
        prime_ratio = prime_count / total_numbers
    else:
        # Default to the proportion of primes in the range if no past data
        prime_ratio = len(prime_numbers) / len(all_numbers)

    # Determine how many primes to select based on the bias
    primes_to_select = round(total_needed * prime_ratio)
    non_primes_to_select = total_needed - primes_to_select

    # Adjust if not enough primes or non-primes are available
    primes_to_select = min(primes_to_select, len(prime_numbers))
    non_primes_to_select = min(non_primes_to_select, len(non_prime_numbers))

    # Randomly select the prime numbers
    selected_numbers = []
    if primes_to_select > 0:
        selected_numbers.extend(random.sample(prime_numbers, primes_to_select))

    # Randomly select the non-prime numbers
    if non_primes_to_select > 0:
        selected_numbers.extend(random.sample(non_prime_numbers, non_primes_to_select))

    # Ensure the total number of selected numbers matches the required amount
    if len(selected_numbers) < total_needed:
        remaining_numbers = list(set(all_numbers) - set(selected_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers.extend(remaining_numbers[:total_needed - len(selected_numbers)])
    elif len(selected_numbers) > total_needed:
        selected_numbers = selected_numbers[:total_needed]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
