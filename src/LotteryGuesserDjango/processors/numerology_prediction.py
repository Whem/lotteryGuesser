# numerology_prediction.py
import datetime
import random
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using numerology prediction.
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
    """Generate a set of numbers using numerology prediction."""
    # Get past draws for pattern analysis
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:50])

    # Extract appropriate number sets from past draws
    if is_main:
        past_numbers = [draw.lottery_type_number for draw in past_draws
                        if isinstance(draw.lottery_type_number, list)]
    else:
        past_numbers = [draw.additional_numbers for draw in past_draws
                        if hasattr(draw, 'additional_numbers') and
                        isinstance(draw.additional_numbers, list)]

    # Get current date for numerological calculations
    current_date = datetime.datetime.now()

    # Generate numbers using multiple numerological methods
    numerology_numbers = generate_numerology_numbers(
        current_date,
        min_num,
        max_num,
        required_numbers,
        past_numbers
    )

    return sorted(numerology_numbers)


def get_life_path_number(date: datetime.datetime) -> int:
    """Calculate life path number from a date."""
    date_str = date.strftime('%Y%m%d')
    total = sum(int(digit) for digit in date_str if digit.isdigit())
    while total > 9:
        total = sum(int(digit) for digit in str(total))
    return total


def get_destiny_number(date: datetime.datetime) -> int:
    """Calculate destiny number based on reduced month and day."""
    month_day = date.strftime('%m%d')
    total = sum(int(digit) for digit in month_day if digit.isdigit())
    while total > 9:
        total = sum(int(digit) for digit in str(total))
    return total


def get_cycle_number(date: datetime.datetime) -> int:
    """Calculate personal year cycle number."""
    year_number = sum(int(digit) for digit in str(date.year))
    month_number = date.month
    day_number = date.day
    total = year_number + month_number + day_number
    while total > 9:
        total = sum(int(digit) for digit in str(total))
    return total


def analyze_number_patterns(past_numbers: List[List[int]], min_num: int, max_num: int) -> Dict[int, float]:
    """Analyze patterns in past numbers through numerological lens."""
    pattern_scores = {num: 0.0 for num in range(min_num, max_num + 1)}

    if not past_numbers:
        return pattern_scores

    # Analyze single digit sums of winning combinations
    for draw in past_numbers:
        draw_sum = sum(int(digit) for num in draw for digit in str(num))
        while draw_sum > 9:
            draw_sum = sum(int(digit) for digit in str(draw_sum))

        # Numbers that sum to this value get higher scores
        for num in range(min_num, max_num + 1):
            num_sum = sum(int(digit) for digit in str(num))
            while num_sum > 9:
                num_sum = sum(int(digit) for digit in str(num_sum))
            if num_sum == draw_sum:
                pattern_scores[num] += 1

    # Normalize scores
    max_score = max(pattern_scores.values())
    if max_score > 0:
        for num in pattern_scores:
            pattern_scores[num] /= max_score

    return pattern_scores


def generate_numerology_numbers(
        current_date: datetime.datetime,
        min_num: int,
        max_num: int,
        required_numbers: int,
        past_numbers: List[List[int]]
) -> List[int]:
    """Generate numbers using multiple numerological methods."""
    selected_numbers: Set[int] = set()

    # Calculate various numerological numbers
    life_path = get_life_path_number(current_date)
    destiny = get_destiny_number(current_date)
    cycle = get_cycle_number(current_date)

    # Analyze patterns in past numbers
    pattern_scores = analyze_number_patterns(past_numbers, min_num, max_num)

    # Generate candidate numbers using different methods
    candidates = []

    # Method 1: Direct numerological numbers
    for base in [life_path, destiny, cycle]:
        num = ((base * life_path) % (max_num - min_num + 1)) + min_num
        candidates.append(num)

    # Method 2: Numerological combinations
    for i in range(1, 4):
        num = ((life_path * i + destiny) % (max_num - min_num + 1)) + min_num
        candidates.append(num)
        num = ((destiny * i + cycle) % (max_num - min_num + 1)) + min_num
        candidates.append(num)

    # Method 3: Pattern-based numbers
    sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
    pattern_candidates = [num for num, _ in sorted_patterns[:required_numbers]]
    candidates.extend(pattern_candidates)

    # Select unique numbers with preference for those matching patterns
    for num in candidates:
        if min_num <= num <= max_num:
            selected_numbers.add(num)
            if len(selected_numbers) >= required_numbers:
                break

    # Fill remaining slots if needed
    while len(selected_numbers) < required_numbers:
        # Try numbers that sum to life path number first
        for num in range(min_num, max_num + 1):
            num_sum = sum(int(digit) for digit in str(num))
            while num_sum > 9:
                num_sum = sum(int(digit) for digit in str(num_sum))
            if num_sum == life_path and num not in selected_numbers:
                selected_numbers.add(num)
                break
        else:
            # If no matching numbers found, add random number
            while len(selected_numbers) < required_numbers:
                num = random.randint(min_num, max_num)
                if num not in selected_numbers:
                    selected_numbers.add(num)

    return sorted(list(selected_numbers))


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when needed."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))