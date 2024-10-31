# number_chain_prediction.py
from collections import defaultdict
import random
from typing import List, Dict, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using chain prediction.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate a set of numbers using chain prediction."""
    # Get past draws
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:200])

    # Extract appropriate number sets from past draws
    if is_main:
        past_numbers = [draw.lottery_type_number for draw in past_draws
                        if isinstance(draw.lottery_type_number, list)]
    else:
        past_numbers = [draw.additional_numbers for draw in past_draws
                        if hasattr(draw, 'additional_numbers') and
                        isinstance(draw.additional_numbers, list)]

    if not past_numbers:
        return generate_random_numbers(min_num, max_num, required_numbers)

    chain_probabilities = calculate_chain_probabilities(past_numbers)
    predicted_numbers = generate_number_chain(
        chain_probabilities,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(set(predicted_numbers))[:required_numbers]


def calculate_chain_probabilities(past_draws: List[List[int]]) -> Dict[int, Dict[int, float]]:
    """Calculate transition probabilities between numbers."""
    chain_counts = defaultdict(lambda: defaultdict(int))
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            chain_counts[sorted_draw[i]][sorted_draw[i + 1]] += 1

    chain_probabilities = defaultdict(dict)
    for first, nexts in chain_counts.items():
        total = sum(nexts.values())
        for second, count in nexts.items():
            chain_probabilities[first][second] = count / total

    return chain_probabilities


def generate_number_chain(
        chain_probabilities: Dict[int, Dict[int, float]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate a chain of numbers based on transition probabilities."""
    predicted_numbers = []

    # Start with a random key from chain probabilities, or a random number if empty
    if chain_probabilities:
        current = random.choice(list(chain_probabilities.keys()))
    else:
        current = random.randint(min_num, max_num)

    while len(predicted_numbers) < required_numbers:
        predicted_numbers.append(current)

        if current in chain_probabilities and chain_probabilities[current]:
            next_numbers = list(chain_probabilities[current].keys())
            probabilities = list(chain_probabilities[current].values())
            current = random.choices(next_numbers, weights=probabilities)[0]
        else:
            # If no chain data available, pick a random number
            current = random.randint(min_num, max_num)

    return predicted_numbers


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))