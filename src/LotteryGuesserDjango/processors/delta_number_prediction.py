import random
from collections import Counter
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    delta_counter = Counter()
    for draw in past_draws:
        sorted_draw = sorted(draw)
        deltas = [sorted_draw[i + 1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
        delta_counter.update(deltas)

    most_common_deltas = [delta for delta, _ in
                          delta_counter.most_common(lottery_type_instance.pieces_of_draw_numbers * 2)]

    for _ in range(100):  # Try up to 100 times to find a valid sequence
        selected_numbers = generate_sequence(lottery_type_instance, most_common_deltas)
        if len(selected_numbers) == lottery_type_instance.pieces_of_draw_numbers:
            return sorted(selected_numbers)

    # If we couldn't generate a valid sequence, fall back to random selection
    return random_selection(lottery_type_instance)


def generate_sequence(lottery_type_instance: lg_lottery_type, deltas: List[int]) -> set:
    selected_numbers = set()
    start_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
    selected_numbers.add(start_number)

    for delta in random.sample(deltas, len(deltas)):  # Shuffle the deltas for more variety
        next_number = start_number + delta
        if lottery_type_instance.min_number <= next_number <= lottery_type_instance.max_number and next_number not in selected_numbers:
            selected_numbers.add(next_number)
            start_number = next_number
            if len(selected_numbers) == lottery_type_instance.pieces_of_draw_numbers:
                break
        else:
            # If adding this delta doesn't work, try subtracting it
            next_number = start_number - delta
            if lottery_type_instance.min_number <= next_number <= lottery_type_instance.max_number and next_number not in selected_numbers:
                selected_numbers.add(next_number)
                start_number = next_number
                if len(selected_numbers) == lottery_type_instance.pieces_of_draw_numbers:
                    break

    return selected_numbers


def random_selection(lottery_type_instance: lg_lottery_type) -> List[int]:
    return random.sample(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
                         lottery_type_instance.pieces_of_draw_numbers)


def analyze_deltas(past_draws: List[Tuple[int, ...]], top_n: int = 5) -> None:
    delta_counter = Counter()
    for draw in past_draws:
        sorted_draw = sorted(draw)
        deltas = [sorted_draw[i + 1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
        delta_counter.update(deltas)

    print(f"Top {top_n} most common deltas:")
    for delta, count in delta_counter.most_common(top_n):
        print(f"Delta {delta} occurred {count} times")

# Example usage of the analysis function:
# past_draws = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True))
# analyze_deltas(past_draws)