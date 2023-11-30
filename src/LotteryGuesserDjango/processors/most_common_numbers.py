#most_common_numbers.py
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type):
    queryset = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).values_list('lottery_type_number',
                                                                                              flat=True)

    # Flatten the list of lists into a single list of numbers
    all_numbers = [number for numbers in queryset for number in numbers]

    # Count occurrences of each number
    number_counter = Counter(all_numbers)

    # Find the five most common numbers
    most_common_numbers = [num for num, count in number_counter.most_common(5)]

    return most_common_numbers