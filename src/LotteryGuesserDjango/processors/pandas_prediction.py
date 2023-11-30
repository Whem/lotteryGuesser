import random

import pandas as pd

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type):


    # Assuming df is your DataFrame with historical lottery data
    # Each row in df represents a draw, with columns like 'number1', 'number2', etc.

    frequency_matrix = {}
    # Querying the database for only the lottery numbers
    queryset = lg_lottery_winner_number.objects.all().values_list('lottery_type_number', flat=True)

    # Convert each list of numbers into a separate row in the DataFrame
    df = pd.DataFrame([numbers for numbers in queryset])
    # Populate frequency matrix
    for index in range(len(df) - 1):
        current_row = df.iloc[index]
        next_row = df.iloc[index + 1]

        for i in range(lottery_type.pieces_of_draw_numbers):
            current_number = current_row[i]
            frequency_matrix.setdefault(current_number, {})

            for j in range(lottery_type.pieces_of_draw_numbers):
                next_number = next_row[j]
                frequency_matrix[current_number][next_number] = frequency_matrix[current_number].get(next_number, 0) + 1

        # Predicting
    min_number = lottery_type.min_number
    max_number = lottery_type.max_number

    # Predicting
    last_draw = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).last()
    if last_draw:
        last_numbers = last_draw.lottery_type_number
        predicted_numbers = []

        for number in last_numbers:
            if len(predicted_numbers) < lottery_type.pieces_of_draw_numbers:
                next_number = max(frequency_matrix.get(number, {}), key=frequency_matrix[number].get, default=None)
                if next_number and min_number <= next_number <= max_number and next_number not in predicted_numbers:
                    predicted_numbers.append(next_number)

        # Ensure the list has the correct number of elements
        while len(predicted_numbers) < lottery_type.pieces_of_draw_numbers:
            random_number = random.randint(min_number, max_number)
            if random_number not in predicted_numbers:
                predicted_numbers.append(random_number)

        return sorted(predicted_numbers)