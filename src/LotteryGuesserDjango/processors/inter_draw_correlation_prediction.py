# inter_draw_correlation_prediction.py
from collections import Counter
from typing import List, Tuple, Set
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers based on inter-draw correlation analysis.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return sorted(main_numbers), sorted(additional_numbers)


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    required_numbers: int,
    is_main: bool
) -> List[int]:
    """
    Generate a set of numbers using inter-draw correlation analysis.
    """
    # Retrieve past draws
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Extract numbers based on whether they are main or additional
    past_draws = []
    for draw in past_draws_queryset:
        if isinstance(draw, list):
            if is_main:
                numbers = [num for num in draw if min_num <= num <= max_num]
            else:
                # Assuming additional numbers are stored separately
                numbers = [num for num in draw if min_num <= num <= max_num]
            if numbers:
                past_draws.append(numbers)

    if not past_draws:
        # If no past draws, return random numbers
        return random.sample(range(min_num, max_num + 1), required_numbers)

    # Calculate number correlations
    number_correlations = calculate_correlations(past_draws)

    predicted_numbers = set()
    for pair, _ in number_correlations.most_common():
        predicted_numbers.update(pair)
        if len(predicted_numbers) >= required_numbers:
            break

    # Adjust the set size if needed
    predicted_numbers = adjust_numbers(predicted_numbers, required_numbers, min_num, max_num)

    # Ensure all numbers are standard Python ints
    predicted_numbers = [int(n) for n in predicted_numbers]

    return predicted_numbers


def calculate_correlations(past_draws: List[List[int]]) -> Counter:
    """
    Calculate correlations between numbers in consecutive draws.
    """
    correlations = Counter()
    for i in range(len(past_draws) - 1):
        current_draw = past_draws[i]
        next_draw = past_draws[i + 1]
        for number in current_draw:
            for next_number in next_draw:
                correlations[(number, next_number)] += 1
    return correlations


def adjust_numbers(
    numbers: Set[int],
    required_numbers: int,
    min_num: int,
    max_num: int
) -> List[int]:
    """
    Adjust the set of numbers to match the required count.
    """
    numbers = set(numbers)
    # Remove extra numbers randomly
    while len(numbers) > required_numbers:
        numbers.remove(random.choice(list(numbers)))

    # Fill missing numbers with random selections
    while len(numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        if new_number not in numbers:
            numbers.add(new_number)

    return list(numbers)
