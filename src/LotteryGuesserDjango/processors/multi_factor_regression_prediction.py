import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 20:
        return sorted(np.random.choice(range(lottery_type_instance.min_number,
                                             lottery_type_instance.max_number + 1),
                                       lottery_type_instance.pieces_of_draw_numbers,
                                       replace=False))

    X = []
    y = []
    for i in range(len(past_draws) - 1):
        X.append([
            np.mean(past_draws[i]),
            np.std(past_draws[i]),
            max(past_draws[i]),
            min(past_draws[i]),
            i  # time factor
        ])
        y.append(past_draws[i+1])

    model = LinearRegression()
    model.fit(X, y)

    last_draw = past_draws[0]
    prediction_input = [
        np.mean(last_draw),
        np.std(last_draw),
        max(last_draw),
        min(last_draw),
        len(past_draws)  # current time
    ]

    prediction = model.predict([prediction_input])[0]
    predicted_numbers = [int(round(num)) for num in prediction]
    predicted_numbers = [num for num in predicted_numbers
                         if lottery_type_instance.min_number <= num <= lottery_type_instance.max_number]

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = np.random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number + 1)
        if new_number not in predicted_numbers:
            predicted_numbers.append(new_number)

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]