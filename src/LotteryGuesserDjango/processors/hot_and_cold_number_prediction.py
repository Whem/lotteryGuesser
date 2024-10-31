# hot_and_cold_number_prediction.py
from collections import Counter
from typing import List, Set, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Hot/cold number predictor for combined lottery types."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

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
        is_main: bool,
        hot_ratio: float = 0.6
) -> List[int]:
    """Generate numbers using hot/cold analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)
    number_counter = Counter(num for draw in past_draws for num in draw)

    if not number_counter:
        return random_selection(min_num, max_num, required_numbers)

    selected_numbers = select_hot_cold_numbers(
        number_counter,
        min_num,
        max_num,
        required_numbers,
        hot_ratio
    )

    fill_missing_numbers(selected_numbers, min_num, max_num, required_numbers)

    return sorted(selected_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def select_hot_cold_numbers(
        number_counter: Counter,
        min_num: int,
        max_num: int,
        required_numbers: int,
        hot_ratio: float
) -> Set[int]:
    """Select numbers based on hot/cold frequency."""
    hot_count = int(required_numbers * hot_ratio)
    cold_count = required_numbers - hot_count

    hot_numbers = set(num for num, _ in number_counter.most_common(hot_count))
    cold_numbers = set(
        num for num, _ in number_counter.most_common()[:-cold_count - 1:-1]
        if min_num <= num <= max_num
    )

    return hot_numbers.union(cold_numbers)


def fill_missing_numbers(
        numbers: Set[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> None:
    """Fill missing numbers to reach required count."""
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))


def random_selection(min_num: int, max_num: int, count: int) -> List[int]:
    """Generate random number selection."""
    return sorted(random.sample(range(min_num, max_num + 1), count))