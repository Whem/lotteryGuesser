import random
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers based on prime position prediction.

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
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Initialize counters for each position
    position_prime_counts = [0] * total_numbers
    position_total_counts = [0] * total_numbers

    for draw in past_draws:
        if not isinstance(draw, list):
            continue
        if len(draw) != total_numbers:
            continue  # Skip draws that don't have the expected number of numbers
        for idx, number in enumerate(draw):
            if isinstance(number, int) and min_num <= number <= max_num:
                position_total_counts[idx] += 1
                if is_prime(number):
                    position_prime_counts[idx] += 1

    # Determine for each position whether to place a prime or non-prime number
    position_is_prime = []
    for prime_count, total_count in zip(position_prime_counts, position_total_counts):
        if total_count == 0:
            position_is_prime.append(False)  # Default to non-prime if no data
        else:
            prime_ratio = prime_count / total_count
            # If primes appear more than 50% in this position, set as prime
            position_is_prime.append(prime_ratio > 0.5)

    # Generate numbers based on position prime status
    primes = [n for n in range(min_num, max_num + 1) if is_prime(n)]
    non_primes = [n for n in range(min_num, max_num + 1) if not is_prime(n)]

    selected_numbers = []
    for is_prime_position in position_is_prime:
        if is_prime_position and primes:
            number = random.choice(primes)
            primes.remove(number)
            selected_numbers.append(number)
        elif not is_prime_position and non_primes:
            number = random.choice(non_primes)
            non_primes.remove(number)
            selected_numbers.append(number)
        elif primes:
            # Fallback if no non-primes are left
            number = random.choice(primes)
            primes.remove(number)
            selected_numbers.append(number)
        elif non_primes:
            # Fallback if no primes are left
            number = random.choice(non_primes)
            non_primes.remove(number)
            selected_numbers.append(number)

    # Ensure unique numbers
    selected_numbers = list(set(selected_numbers))

    # If not enough numbers, fill with remaining numbers
    all_numbers = set(range(min_num, max_num + 1))
    remaining_numbers = list(all_numbers - set(selected_numbers))
    random.shuffle(remaining_numbers)
    while len(selected_numbers) < total_numbers and remaining_numbers:
        selected_numbers.append(remaining_numbers.pop())

    # Trim the list if we have too many numbers
    selected_numbers = selected_numbers[:total_numbers]

    # Sort and return the selected numbers
    selected_numbers.sort()
    return selected_numbers
