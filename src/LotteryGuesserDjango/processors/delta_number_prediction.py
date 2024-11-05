import random
from collections import Counter
from typing import List, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using delta pattern analysis.
    Returns both main numbers and additional numbers (if applicable).

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing:
        - main_numbers: List of main lottery numbers
        - additional_numbers: List of additional numbers (empty if not applicable)
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        'lottery_type_number',
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            'additional_numbers',
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count
        )

    return main_numbers, additional_numbers


def generate_numbers(
        lottery_type_instance: lg_lottery_type,
        number_field: str,
        min_num: int,
        max_num: int,
        total_numbers: int
) -> List[int]:
    """Generate numbers using delta pattern analysis."""
    try:
        # Get past draws
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')

        past_draws = []
        for draw in past_draws_queryset:
            numbers = getattr(draw, number_field, None)
            if isinstance(numbers, list) and len(numbers) == total_numbers:
                past_draws.append(numbers)

        if not past_draws:
            return random_selection(min_num, max_num, total_numbers)

        # Calculate delta patterns
        delta_counter = calculate_deltas(past_draws)

        # Get most common deltas
        most_common_deltas = [delta for delta, _ in
                              delta_counter.most_common(total_numbers * 2)]

        # Try to generate sequence using deltas
        for _ in range(100):  # Try up to 100 times
            selected_numbers = generate_sequence_from_deltas(
                min_num, max_num, total_numbers, most_common_deltas
            )
            if len(selected_numbers) == total_numbers:
                return sorted(list(selected_numbers))

        # Fallback to random selection if no valid sequence found
        return random_selection(min_num, max_num, total_numbers)

    except Exception as e:
        print(f"Error in generate_numbers: {str(e)}")
        return random_selection(min_num, max_num, total_numbers)


def calculate_deltas(past_draws: List[List[int]]) -> Counter:
    """Calculate delta patterns from past draws."""
    delta_counter = Counter()
    try:
        for draw in past_draws:
            sorted_draw = sorted(draw)
            deltas = [sorted_draw[i + 1] - sorted_draw[i]
                      for i in range(len(sorted_draw) - 1)]
            delta_counter.update(deltas)
    except Exception as e:
        print(f"Error in calculate_deltas: {str(e)}")
    return delta_counter


def generate_sequence_from_deltas(
        min_num: int,
        max_num: int,
        total_numbers: int,
        deltas: List[int]
) -> Set[int]:
    """Generate a sequence of numbers using delta patterns."""
    selected_numbers = set()
    try:
        start_number = random.randint(min_num, max_num)
        selected_numbers.add(start_number)

        shuffled_deltas = random.sample(deltas, len(deltas))
        current_number = start_number

        for delta in shuffled_deltas:
            if len(selected_numbers) >= total_numbers:
                break

            # Try adding delta
            next_number = current_number + delta
            if min_num <= next_number <= max_num and next_number not in selected_numbers:
                selected_numbers.add(next_number)
                current_number = next_number
                continue

            # Try subtracting delta if adding didn't work
            next_number = current_number - delta
            if min_num <= next_number <= max_num and next_number not in selected_numbers:
                selected_numbers.add(next_number)
                current_number = next_number

    except Exception as e:
        print(f"Error in generate_sequence_from_deltas: {str(e)}")

    return selected_numbers


def random_selection(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """Generate random numbers within the specified range."""
    try:
        if max_num < min_num or total_numbers <= 0:
            return []

        numbers = set()
        available_range = list(range(min_num, max_num + 1))

        if total_numbers > len(available_range):
            total_numbers = len(available_range)

        while len(numbers) < total_numbers:
            numbers.add(random.choice(available_range))

        return sorted(list(numbers))
    except Exception as e:
        print(f"Error in random_selection: {str(e)}")
        return list(range(min_num, min(min_num + total_numbers, max_num + 1)))


def analyze_deltas(past_draws: List[List[int]], top_n: int = 5) -> None:
    """Analyze delta patterns in past draws."""
    try:
        delta_counter = Counter()
        for draw in past_draws:
            sorted_draw = sorted(draw)
            deltas = [sorted_draw[i + 1] - sorted_draw[i]
                      for i in range(len(sorted_draw) - 1)]
            delta_counter.update(deltas)

        print(f"\nTop {top_n} most common deltas:")
        for delta, count in delta_counter.most_common(top_n):
            print(f"Delta {delta} occurred {count} times")

    except Exception as e:
        print(f"Error in analyze_deltas: {str(e)}")