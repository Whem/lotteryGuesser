# adaptive_hybrid_prediction.py
import numpy as np
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Adaptive hybrid prediction algorithm for both simple and combined lottery types.
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
    """Generate a set of numbers using hybrid prediction methods."""
    # Get past draws
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])

    # Extract appropriate number sets from past draws
    if is_main:
        past_numbers = [draw.lottery_type_number for draw in past_draws
                        if isinstance(draw.lottery_type_number, list)]
    else:
        past_numbers = [draw.additional_numbers for draw in past_draws
                        if hasattr(draw, 'additional_numbers') and
                        isinstance(draw.additional_numbers, list)]

    if len(past_numbers) < 20:
        return sorted(np.random.choice(
            range(min_num, max_num + 1),
            required_numbers,
            replace=False
        ).tolist())

    # Define prediction methods
    methods = [
        # Method 1: Pure random selection
        lambda: np.random.choice(
            range(min_num, max_num + 1),
            required_numbers,
            replace=False
        ).tolist(),

        # Method 2: Use most recent draw
        lambda: (sorted(past_numbers[0])[:required_numbers]
                 if past_numbers and len(past_numbers[0]) >= required_numbers
                 else np.random.choice(range(min_num, max_num + 1),
                                       required_numbers,
                                       replace=False).tolist()),

        # Method 3: Historical frequency-based selection
        lambda: np.random.choice(
            np.unique([num for draw in past_numbers for num in draw]),
            required_numbers,
            replace=False
        ).tolist(),

        # Method 4: Mean-based selection
        lambda: generate_mean_based_numbers(
            past_numbers,
            min_num,
            max_num,
            required_numbers
        ),

        # Method 5: Pattern-based selection
        lambda: generate_pattern_based_numbers(
            past_numbers,
            min_num,
            max_num,
            required_numbers
        )
    ]

    # Initialize weights
    weights = [1.0 / len(methods)] * len(methods)

    # Generate predictions from each method
    predictions = []
    for method in methods:
        try:
            pred = method()
            if len(pred) == required_numbers:
                predictions.append(pred)
            else:
                # Fallback to random if method fails
                predictions.append(np.random.choice(
                    range(min_num, max_num + 1),
                    required_numbers,
                    replace=False
                ).tolist())
        except Exception as e:
            print(f"Method error: {e}")
            # Fallback to random
            predictions.append(np.random.choice(
                range(min_num, max_num + 1),
                required_numbers,
                replace=False
            ).tolist())

    # Evaluate recent performance
    if past_numbers:
        recent_draw = past_numbers[0]
        performances = [len(set(pred) & set(recent_draw)) for pred in predictions]

        # Update weights based on performance
        total_performance = sum(performances)
        if total_performance > 0:
            weights = [p / total_performance for p in performances]
        else:
            weights = [1.0 / len(methods)] * len(methods)

    # Generate final prediction using weighted selection
    final_prediction = set()
    attempts = 0
    max_attempts = 1000  # Prevent infinite loops

    while len(final_prediction) < required_numbers and attempts < max_attempts:
        method_index = int(np.random.choice(len(methods), p=weights))
        number = int(np.random.choice(predictions[method_index]))

        if min_num <= number <= max_num and number not in final_prediction:
            final_prediction.add(number)

        attempts += 1

    # If we still don't have enough numbers, fill with random ones
    while len(final_prediction) < required_numbers:
        number = np.random.randint(min_num, max_num + 1)
        if number not in final_prediction:
            final_prediction.add(number)

    return sorted(list(final_prediction))


def generate_mean_based_numbers(
        past_numbers: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on historical means."""
    if not past_numbers:
        return np.random.choice(range(min_num, max_num + 1),
                                required_numbers,
                                replace=False).tolist()

    # Calculate mean for each position
    position_means = []
    for pos in range(required_numbers):
        position_numbers = [draw[pos] for draw in past_numbers if len(draw) > pos]
        if position_numbers:
            position_means.append(int(np.mean(position_numbers)))
        else:
            position_means.append(np.random.randint(min_num, max_num + 1))

    # Adjust means to ensure unique numbers within bounds
    result = set()
    for mean in position_means:
        number = mean
        attempts = 0
        while (number in result or number < min_num or number > max_num) and attempts < 100:
            offset = np.random.randint(-5, 6)  # Random adjustment
            number = mean + offset
            attempts += 1

        if attempts >= 100 or number < min_num or number > max_num:
            # If we failed to find a valid number, pick a random one
            available = set(range(min_num, max_num + 1)) - result
            if available:
                number = np.random.choice(list(available))
            else:
                number = np.random.randint(min_num, max_num + 1)

        result.add(number)

    return sorted(list(result))


def generate_pattern_based_numbers(
        past_numbers: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate numbers based on common patterns in past draws."""
    if not past_numbers or len(past_numbers) < 2:
        return np.random.choice(range(min_num, max_num + 1),
                                required_numbers,
                                replace=False).tolist()

    # Calculate common differences between consecutive numbers
    differences = []
    for draw in past_numbers:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            differences.append(sorted_draw[i + 1] - sorted_draw[i])

    # Use median difference to generate new numbers
    median_diff = int(np.median(differences)) if differences else 1

    # Start with a random first number
    result = set()
    start_num = np.random.randint(min_num, max_num - (required_numbers - 1) * median_diff)
    result.add(start_num)

    # Generate subsequent numbers using the pattern
    current = start_num
    while len(result) < required_numbers:
        next_num = current + median_diff
        if next_num > max_num:
            # If we exceed the maximum, start a new sequence
            available = set(range(min_num, max_num + 1)) - result
            if available:
                next_num = np.random.choice(list(available))
            else:
                next_num = np.random.randint(min_num, max_num + 1)
        current = next_num
        result.add(current)

    return sorted(list(result))