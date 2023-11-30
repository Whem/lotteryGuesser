from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    pattern_counter = Counter()
    pattern_length = 3  # Example pattern length

    for i in range(len(past_draws) - pattern_length):
        pattern = tuple(past_draws[i:i + pattern_length])
        pattern_counter[pattern] += 1

    most_common_patterns = [pattern for pattern, _ in pattern_counter.most_common()]
    selected_numbers = set()
    for pattern in most_common_patterns:
        selected_numbers.update(pattern)
        if len(selected_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    return sorted(selected_numbers)[:lottery_type_instance.pieces_of_draw_numbers]