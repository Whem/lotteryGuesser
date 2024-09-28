from typing import List, Dict
from collections import defaultdict
from statistics import mean
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type: lg_lottery_type) -> List[int]:
    queryset = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).values_list('lottery_type_number',
                                                                                              flat=True)

    intervals = defaultdict(list)
    last_occurrence = {}
    current_draw = 0

    for numbers in queryset:
        for number in numbers:
            if number in last_occurrence:
                intervals[number].append(current_draw - last_occurrence[number])
            last_occurrence[number] = current_draw
        current_draw += 1

    average_intervals = {number: mean(interval_list) for number, interval_list in intervals.items()}

    predicted_numbers = sorted(average_intervals, key=lambda x: (average_intervals[x], -x))[
                        :lottery_type.pieces_of_draw_numbers]

    return predicted_numbers