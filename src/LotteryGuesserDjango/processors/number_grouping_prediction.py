from collections import defaultdict, Counter
import random
from typing import List, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    group_patterns = analyze_group_patterns(past_draws, lottery_type_instance)
    predicted_numbers = generate_numbers_from_patterns(group_patterns, lottery_type_instance)

    return sorted(predicted_numbers)


def analyze_group_patterns(past_draws: List[List[int]], lottery_type_instance: lg_lottery_type) -> Dict[str, Counter]:
    group_size = (lottery_type_instance.max_number - lottery_type_instance.min_number + 1) // 3
    groups = {
        'low': range(lottery_type_instance.min_number, lottery_type_instance.min_number + group_size),
        'mid': range(lottery_type_instance.min_number + group_size, lottery_type_instance.min_number + 2 * group_size),
        'high': range(lottery_type_instance.min_number + 2 * group_size, lottery_type_instance.max_number + 1)
    }

    patterns = Counter()
    for draw in past_draws:
        pattern = ''.join(sorted([get_group(num, groups) for num in draw]))
        patterns[pattern] += 1

    return patterns


def get_group(number: int, groups: Dict[str, range]) -> str:
    for group, range_ in groups.items():
        if number in range_:
            return group[0]  # Return first letter of group name
    return 'x'  # Should never happen


def generate_numbers_from_patterns(patterns: Counter, lottery_type_instance: lg_lottery_type) -> List[int]:
    group_size = (lottery_type_instance.max_number - lottery_type_instance.min_number + 1) // 3
    groups = {
        'l': range(lottery_type_instance.min_number, lottery_type_instance.min_number + group_size),
        'm': range(lottery_type_instance.min_number + group_size, lottery_type_instance.min_number + 2 * group_size),
        'h': range(lottery_type_instance.min_number + 2 * group_size, lottery_type_instance.max_number + 1)
    }

    most_common_pattern = patterns.most_common(1)[0][0]
    predicted_numbers = []

    for group in most_common_pattern:
        predicted_numbers.append(random.choice(list(groups[group])))

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.append(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return list(set(predicted_numbers))[:lottery_type_instance.pieces_of_draw_numbers]