# entropy_based_prediction.py
import numpy as np
from scipy.stats import entropy
from typing import List, Tuple, Set, Dict
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random  # Make sure to import random

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Entropy-based predictor for combined lottery types.
    Returns (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using entropy-based analysis."""
    try:
        # Get historical data
        past_draws = get_historical_data(lottery_type_instance, is_main)

        if len(past_draws) < 20:
            return random_number_set(min_num, max_num, required_numbers)

        # Calculate entropy and probabilities
        entropy_value, probabilities = calculate_entropy(
            past_draws,
            min_num,
            max_num
        )

        # Generate predictions
        predicted_numbers = generate_predictions(
            entropy_value,
            probabilities,
            min_num,
            max_num,
            required_numbers
        )

        # Convert all numbers to standard Python ints
        predicted_numbers = [int(n) for n in predicted_numbers]

        return sorted(predicted_numbers)

    except Exception as e:
        print(f"Error in entropy prediction: {str(e)}")
        return random_number_set(min_num, max_num, required_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def calculate_entropy(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Tuple[float, Dict[int, float]]:
    """Calculate entropy and probabilities from historical data."""
    # Flatten past draws and count frequencies
    number_counts = Counter(
        num for draw in past_draws
        for num in draw
        if min_num <= num <= max_num
    )

    # Initialize probabilities for all possible numbers
    probabilities = {num: 0.0 for num in range(min_num, max_num + 1)}
    total_numbers = sum(number_counts.values())

    # Calculate probabilities
    if total_numbers > 0:
        for num, count in number_counts.items():
            probabilities[num] = count / total_numbers

    # Add small probability for unseen numbers
    min_prob = 1.0 / (total_numbers + len(probabilities))
    for num in probabilities:
        if probabilities[num] == 0:
            probabilities[num] = min_prob

    # Normalize probabilities
    prob_sum = sum(probabilities.values())
    probabilities = {
        num: prob / prob_sum
        for num, prob in probabilities.items()
    }

    # Calculate entropy
    prob_values = list(probabilities.values())
    entropy_value = float(entropy(prob_values))

    return entropy_value, probabilities


def generate_predictions(
        entropy_value: float,
        probabilities: Dict[int, float],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions based on entropy value."""
    predicted_numbers = set()
    all_numbers = list(range(min_num, max_num + 1))
    prob_list = [probabilities[num] for num in all_numbers]

    # High entropy threshold increased for more stability
    high_entropy_threshold = 0.95

    while len(predicted_numbers) < required_numbers:
        try:
            if entropy_value > high_entropy_threshold:
                # High entropy - more random selection
                remaining = required_numbers - len(predicted_numbers)
                available = list(set(all_numbers) - predicted_numbers)
                new_numbers = np.random.choice(
                    available,
                    size=remaining,
                    replace=False
                )
                # Ensure numbers are standard ints
                predicted_numbers.update(int(n) for n in new_numbers)
            else:
                # Lower entropy - use probability-based selection
                new_number = np.random.choice(
                    all_numbers,
                    p=prob_list
                )
                # Ensure number is a standard int
                new_number = int(new_number)
                if new_number not in predicted_numbers:
                    predicted_numbers.add(new_number)

        except Exception as e:
            print(f"Error in prediction generation: {str(e)}")
            # Fallback to random selection
            remaining = required_numbers - len(predicted_numbers)
            available = list(set(range(min_num, max_num + 1)) - predicted_numbers)
            if available:
                new_numbers = random.sample(available, min(remaining, len(available)))
                predicted_numbers.update(new_numbers)

    return list(predicted_numbers)


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    numbers = random.sample(range(min_num, max_num + 1), required_numbers)
    return [int(n) for n in numbers]
