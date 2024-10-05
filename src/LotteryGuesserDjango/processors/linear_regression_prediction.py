# linear_regression_prediction.py

import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance):


    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_draws = [
        [int(num) for num in draw] for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 20:
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    X = np.arange(len(past_draws)).reshape(-1, 1)

    models = []
    for num_position in range(total_numbers):
        y = np.array([draw[num_position] for draw in past_draws])
        model = LinearRegression()
        model.fit(X, y)
        models.append(model)

    next_index = np.array([[len(past_draws)]])
    predicted_numbers = []
    for idx, model in enumerate(models):
        pred = model.predict(next_index)[0]
        pred_int = int(round(pred))
        pred_int = max(min_num, min(pred_int, max_num))
        if pred_int not in predicted_numbers:
            predicted_numbers.append(pred_int)
        if len(predicted_numbers) == total_numbers:
            break

    if len(predicted_numbers) < total_numbers:
        all_numbers = [num for draw in past_draws for num in draw]
        number_counts = Counter(all_numbers)
        most_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]
        for num in most_common_numbers:
            predicted_numbers.append(int(num))
            if len(predicted_numbers) == total_numbers:
                break

    if len(predicted_numbers) < total_numbers:
        for num in range(min_num, max_num + 1):
            if num not in predicted_numbers:
                predicted_numbers.append(int(num))
                if len(predicted_numbers) == total_numbers:
                    break


    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers = [int(num) for num in predicted_numbers]
    predicted_numbers.sort()


    return predicted_numbers
