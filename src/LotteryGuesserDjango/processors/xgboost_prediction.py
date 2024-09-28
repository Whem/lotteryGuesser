# xgboost_prediction.py

import random
import numpy as np
import xgboost as xgb
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using XGBoost regression.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Parameters
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_draws = [draw for draw in past_draws_queryset if isinstance(draw, list) and len(draw) == total_numbers]

    if len(past_draws) < 10:
        # Not enough data to train the model
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Prepare training data
    X = []
    y = []
    for i in range(len(past_draws) - 1):
        X.append(past_draws[i])
        y.append(past_draws[i + 1])

    X = np.array(X)
    y = np.array(y)

    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    # Use the last draw as input for prediction
    last_draw = np.array(past_draws[-1]).reshape(1, -1)
    prediction = model.predict(last_draw)[0]

    # Extract predicted numbers
    predicted_numbers = [int(round(num)) for num in prediction]
    predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]
    predicted_numbers = list(set(predicted_numbers))

    # If we have enough predicted numbers
    if len(predicted_numbers) >= total_numbers:
        selected_numbers = predicted_numbers[:total_numbers]
    else:
        # Fill the remaining numbers randomly
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers = predicted_numbers + remaining_numbers[:total_numbers - len(predicted_numbers)]

    # Ensure the correct number of numbers
    selected_numbers = selected_numbers[:total_numbers]
    selected_numbers.sort()
    return selected_numbers
