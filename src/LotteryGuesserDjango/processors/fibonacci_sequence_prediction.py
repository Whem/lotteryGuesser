import random
from typing import List, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    fibonacci_numbers = generate_fibonacci_sequence(lottery_type_instance.max_number)
    near_fibonacci_numbers = generate_near_fibonacci_numbers(fibonacci_numbers, lottery_type_instance)

    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    frequency = {num: 0 for num in near_fibonacci_numbers}

    for draw in past_draws:
        for number in draw:
            if number in frequency:
                frequency[number] += 1

    # Sort near_fibonacci_numbers by their frequency in past draws
    sorted_numbers = sorted(near_fibonacci_numbers, key=lambda x: frequency[x], reverse=True)

    # Select the top most frequent numbers, then fill with random selections if needed
    predicted_numbers = set(sorted_numbers[:lottery_type_instance.pieces_of_draw_numbers])
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        number = random.choice(sorted_numbers)
        predicted_numbers.add(number)

    return sorted(predicted_numbers)


def generate_fibonacci_sequence(max_number: int) -> List[int]:
    sequence = [0, 1]
    while sequence[-1] < max_number:
        next_number = sequence[-1] + sequence[-2]
        if next_number > max_number:
            break
        sequence.append(next_number)
    return sequence[2:]  # Exclude 0 and 1


def generate_near_fibonacci_numbers(fibonacci_numbers: List[int], lottery_type_instance: lg_lottery_type) -> Set[int]:
    near_fibonacci = set()
    for fib in fibonacci_numbers:
        for i in range(-2, 3):  # Include numbers up to 2 away from Fibonacci numbers
            number = fib + i
            if lottery_type_instance.min_number <= number <= lottery_type_instance.max_number:
                near_fibonacci.add(number)
    return near_fibonacci


def analyze_fibonacci_patterns(past_draws: List[List[int]], lottery_type_instance: lg_lottery_type) -> None:
    fibonacci_numbers = generate_fibonacci_sequence(lottery_type_instance.max_number)
    near_fibonacci_numbers = generate_near_fibonacci_numbers(fibonacci_numbers, lottery_type_instance)

    total_numbers = sum(len(draw) for draw in past_draws)
    near_fibonacci_count = sum(1 for draw in past_draws for num in draw if num in near_fibonacci_numbers)

    print(f"Fibonacci analysis:")
    print(f"Total numbers in past draws: {total_numbers}")
    print(f"Numbers near Fibonacci sequence: {near_fibonacci_count}")
    print(f"Percentage of numbers near Fibonacci: {near_fibonacci_count / total_numbers * 100:.2f}%")