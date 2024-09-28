import random
import pandas as pd
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type: lg_lottery_type) -> List[int]:
    queryset = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).values_list('lottery_type_number',
                                                                                              flat=True)

    if not queryset:
        return random.sample(range(lottery_type.min_number, lottery_type.max_number + 1),
                             lottery_type.pieces_of_draw_numbers)

    df = pd.DataFrame(list(queryset))

    frequency_matrix = {}
    for index in range(len(df) - 1):
        current_row = df.iloc[index]
        next_row = df.iloc[index + 1]

        for i, current_number in enumerate(current_row):
            frequency_matrix.setdefault(current_number, {})
            for next_number in next_row:
                frequency_matrix[current_number][next_number] = frequency_matrix[current_number].get(next_number, 0) + 1

    last_numbers = df.iloc[-1].tolist()
    predicted_numbers = []

    for number in last_numbers:
        if len(predicted_numbers) < lottery_type.pieces_of_draw_numbers:
            if number in frequency_matrix:
                next_number = max(frequency_matrix[number], key=frequency_matrix[number].get)
                if lottery_type.min_number <= next_number <= lottery_type.max_number and next_number not in predicted_numbers:
                    predicted_numbers.append(next_number)

    while len(predicted_numbers) < lottery_type.pieces_of_draw_numbers:
        random_number = random.randint(lottery_type.min_number, lottery_type.max_number)
        if random_number not in predicted_numbers:
            predicted_numbers.append(random_number)

    return sorted(predicted_numbers)